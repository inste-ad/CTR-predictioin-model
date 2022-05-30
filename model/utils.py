import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print(f'epoch {epoch}')
        print(',\n', end='')

    def on_batch_begin(self, batch, logs):
        if batch % 10 == 0:
            print('.', end='')
