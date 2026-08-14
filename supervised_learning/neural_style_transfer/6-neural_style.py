def content_cost(self, content_output):
        """Calculates the content cost for the generated image

        content_output - a tf.Tensor containing the content output for
            the generated image

        Returns: the content cost
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != s:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        return tf.reduce_mean(tf.square(content_output - self.content_feature))
