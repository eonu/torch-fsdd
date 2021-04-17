class TrimSilence:
    """Removes the silence at the beginning and end of the passed audio data.

    .. warning::
        This transformation assumes that the audio **is** normalized.

    Parameters
    ----------
    threshold: float
        The maximum amount of noise that is considered silence.
    """
    def __init__(self, threshold):
        assert 0. <= threshold <= 1.
        self.threshold = threshold

    def __call__(self, x):
        """Applies the transformation.

        Parameters
        ----------
        x: torch.Tensor
            A one-dimensional tensor of WAV audio samples.

        Returns
        -------
        x: :class:`torch:torch.Tensor`
            The original tensor trimmed for silence.
        """
        start, end = 0, 0

        for i, sample in enumerate(x):
            if abs(sample) > self.threshold:
                start = i
                break

        # Reverse the array for trimming the end
        for i, sample in enumerate(x.flip(dims=(0,))):
            if abs(sample) > self.threshold:
                end = len(x) - i
                break

        return x[start:end]