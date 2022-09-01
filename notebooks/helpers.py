import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_chart(history, plot_details, plot_labels, title, labels):
    '''
    Plot the training and validation accuracy/loss

    Args:
        history: The history object returned by the model.fit() method
        plot_details: The list of keys to plot from the history object
        plot_labels: The list of labels for the plot
        title: The title of the plot
        labels: The dictionary of labels for the plot

    Returns:
        None
    '''
    if len(plot_details) > 0:
        plt.plot(
            history.history[plot_details[0]],
            label=plot_labels[0],
            color="#ea9999",
        )
    if len(plot_details) > 1:
        plt.plot(
            history.history[plot_details[1]],
            label=plot_labels[1],
            color="#9fc5e8",
        )
    plt.title(title)
    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.legend(labels["legend_labels"], loc="upper left")
    plt.show()


def run_model(
    inputs,
    outputs,
    train_dataset,
    val_dataset,
    epochs=30,
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
    callback=None,
    plt_show=True,
    plt_title="Model Evaluation",
):
    '''
    Run the model

    Args:
        inputs: The input layer
        outputs: The output layer
        train_dataset: The training dataset
        val_dataset: The validation dataset
        epochs: The number of epochs to train the model
        optimizer: The optimizer to use
        loss: The loss function to use
        metrics: The metrics to use
        callback: The callback to use
        plt_show: Whether to show the plot or not
        plt_title: The title of the plot

    Returns:
        history: The history object returned by the model.fit() method
        model: The model
    '''
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    if callback is None:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
        )
    else:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback,
        )

    if plt_show:
        plot_chart(
            history,
            ["accuracy", "val_accuracy"],
            ["Training Accuracy", "Validation Accuracy"],
            plt_title,
            {
                "x_label": "Epoch",
                "y_label": "Accuracy",
                "legend_labels": ["Training", "Validation"],
            },
        )

        plot_chart(
            history,
            ["loss", "val_loss"],
            ["Training Loss", "Validation Loss"],
            plt_title,
            {
                "x_label": "Epoch",
                "y_label": "Loss",
                "legend_labels": ["Training", "Validation"],
            },
        )

    return history, model


def split_data(dataset, train_size, val_size):
    '''
    Split the dataset into training and validation datasets

    Args:
        dataset: The dataset to split
        train_size: The size of the training dataset
        val_size: The size of the validation dataset

    Returns:
        train_dataset: The training dataset
        val_dataset: The validation dataset
    '''
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE).cache()
    val_dataset = val_dataset.batch(8).prefetch(tf.data.AUTOTUNE).cache()
    test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE).cache()

    return train_dataset, val_dataset, test_dataset
