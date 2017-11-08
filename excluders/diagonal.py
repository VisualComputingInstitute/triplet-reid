class Excluder(object):
    def __init__(self, gallery_fids):
        # Store the gallery data
        self.gallery_fids = gallery_fids

    def __call__(self, query_fids):
        # Only make sure we don't match the exact same image.
        return self.gallery_fids[None] == query_fids[..., None]
