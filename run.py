# run.py


from sklearn.model_selection import train_test_split
import numpy as np

from utils.load_data import load, LoadSpec, make_path

from settings import Config
config = Config(train=True)


def _create_ordinal_labels(tlab):
    # Create a lab lookup for later
    lookup = {i:col for i, col in enumerate(tlab.columns)}  # NOT IN USE

    # Establish an integer vector to dot against the bits in dataframe
    # Note!  This method assumes one lab per row
    vect = np.arange(len(tlab.columns))

    label_ar = tlab.to_numpy()
    nlabels = np.dot(label_ar, vect).astype(int)

    return nlabels


def prepare_data(files, label_type='default'):
    training_labels = files['training_labels'][0]
    training_data = files['training_data'][0]
    test_data = files['test_data'][0]

    if label_type == 'ordinal':
        training_labels = _create_ordinal_labels(training_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        training_data.iloc[:, 1:], training_labels, 
                                        random_state=0)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model_fn = config['MODEL']
    clf = model_fn()
    print('Model Created: ', clf)

    data_dir = config.DATASET_PATH

    files = {
        'training_data': [
            LoadSpec(make_path(data_dir, 'train_values_mod.csv'), index_col='sequence_id'),
            ],
        'training_labels': [
            LoadSpec(make_path(data_dir, 'train_labels.csv'), index_col='sequence_id'),
            ],
        'test_data': [
            LoadSpec(make_path(data_dir, 'test_values.csv'), index_col='sequence_id'),
            ],
    }

    print('Files Specified: \n', files)

    X_train, X_test, y_train, y_test = prepare_data(load(files=files), label_type='ordinal')

    clf.fit(X_train, y_train)
    print('Score: ', clf.score(X_test, y_test))
    

