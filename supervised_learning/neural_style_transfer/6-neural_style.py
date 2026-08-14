def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        parameters:
            content_output [tf.Tensor]:
                contains the content output for the generated image

        returns:
            the content cost
        """
        shape = self.content_feature.shape
        content_valid = (
            isinstance(content_output, (tf.Tensor, tf.Variable)) and
            content_output.shape == shape
        )
        if not content_valid:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(shape))

        return tf.reduce_mean(tf.square(content_output - self.content_feature))
