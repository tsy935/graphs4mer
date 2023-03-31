"""Adapted from https://github.com/helme/ecg_ptbxl_benchmarking"""

import numpy as np
import pandas as pd
import os
import ast
import argparse
import wfdb
import pickle
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tqdm import tqdm


TRAIN_FOLD = 8
VAL_FOLD = 9
TEST_FOLD = 10


def select_data(XX, YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == "diagnostic":
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == "subdiagnostic":
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["subdiagnostic_len"] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == "superdiagnostic":
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["superdiagnostic_len"] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == "form":
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["form_len"] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == "rhythm":
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["rhythm_len"] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == "all":
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        YY["all_scp_len"] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(os.path.join(outputfolder, "mlb.pkl"), "wb") as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb


def load_raw_data_icbeb(df, sampling_rate, path):

    if sampling_rate == 100:
        if os.path.exists(path + "raw100.npy"):
            data = np.load(path + "raw100.npy", allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + "records100/" + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, "raw100.npy"), "wb"), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + "raw500.npy"):
            data = np.load(path + "raw500.npy", allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + "records500/" + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, "raw500.npy"), "wb"), protocol=4)
    return data


def load_dataset(path, sampling_rate):
    # load and convert annotation data
    Y = pd.read_csv(path + "icbeb_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y


def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    with open(os.path.join(outputfolder, "standard_scaler.pkl"), "wb") as ss_file:
        pickle.dump(ss, ss_file)

    return (
        apply_standardizer(X_train, ss),
        apply_standardizer(X_validation, ss),
        apply_standardizer(X_test, ss),
    )


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.asarray(X_tmp, dtype=object)
    return X_tmp


def compute_label_aggregations(df, folder, ctype):

    df["scp_codes_len"] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder + "scp_statements.csv", index_col=0)

    if ctype in ["diagnostic", "subdiagnostic", "superdiagnostic"]:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == "diagnostic":
            df["diagnostic"] = df.scp_codes.apply(aggregate_all_diagnostic)
            df["diagnostic_len"] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == "subdiagnostic":
            df["subdiagnostic"] = df.scp_codes.apply(aggregate_subdiagnostic)
            df["subdiagnostic_len"] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == "superdiagnostic":
            df["superdiagnostic"] = df.scp_codes.apply(aggregate_diagnostic)
            df["superdiagnostic_len"] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == "form":
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        df["form"] = df.scp_codes.apply(aggregate_form)
        df["form_len"] = df.form.apply(lambda x: len(x))
    elif ctype == "rhythm":
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        df["rhythm"] = df.scp_codes.apply(aggregate_rhythm)
        df["rhythm_len"] = df.rhythm.apply(lambda x: len(x))
    elif ctype == "all":
        df["all_scp"] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def main(raw_data_dir, output_dir, task="all", sampling_freq=100):
    data, raw_labels = load_dataset(raw_data_dir, sampling_freq)

    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, raw_data_dir, task)

    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(
        data,
        labels,
        task,
        0,
        output_dir,
    )
    input_shape = data[0].shape

    # 10th fold for testing
    X_test = data[labels.strat_fold == TEST_FOLD]
    y_test = Y[labels.strat_fold == TEST_FOLD]

    # 9th fold for validation
    X_val = data[labels.strat_fold == VAL_FOLD]
    y_val = Y[labels.strat_fold == VAL_FOLD]
    # rest for training
    X_train = data[labels.strat_fold <= TRAIN_FOLD]
    y_train = Y[labels.strat_fold <= TRAIN_FOLD]

    # Standardize signal data
    X_train, X_val, X_test = preprocess_signals(
        X_train,
        X_val,
        X_test,
        output_dir,
    )

    # save data
    X_train.dump(os.path.join(output_dir, "X_train.npy"))
    X_val.dump(os.path.join(output_dir, "X_val.npy"))
    X_test.dump(os.path.join(output_dir, "X_test.npy"))

    y_train.dump(os.path.join(output_dir, "y_train.npy"))
    y_val.dump(os.path.join(output_dir, "y_val.npy"))
    y_test.dump(os.path.join(output_dir, "y_test.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preproc ICBEB.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Full path to raw data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Dir to save preprocessed data.",
    )
    parser.add_argument(
        "--sampling_freq",
        type=int,
        default=100,
        choices=(100, 500),
        help="Sampling frequency.",
    )
    args = parser.parse_args()
    main(
        args.raw_data_dir, args.output_dir, task="all", sampling_freq=args.sampling_freq
    )
