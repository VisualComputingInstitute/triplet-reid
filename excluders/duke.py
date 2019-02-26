import os
import re

import numpy as np


class Excluder:
    """
    In the DukeMTMC-reID evaluation, we need to exclude distractions
    """
    def __init__(self, gallery_fids):
        # Setup a regexp for extracing the PID and camera (CID) form a FID.
        self.regexp = re.compile('(\S+)_c(\d+)_.*')

        # Parse the gallery_set
        self.gallery_pids, self.gallery_cids = self._parse(gallery_fids)

    def __call__(self, query_fids):
        # Extract both the PIDs and CIDs from the query filenames:
        query_pids, query_cids = self._parse(query_fids)

        # Ignore same pid image within the same camera
        cid_matches = self.gallery_cids[None] == query_cids[:,None]
        pid_matches = self.gallery_pids[None] == query_pids[:,None]
        mask = np.logical_and(cid_matches, pid_matches)

        # Remove all "junk" with the -1 pid.
        junk_images = np.repeat(self.gallery_pids[None] == '-1', len(query_pids), 0)
        mask = np.logical_or(mask, junk_images)

        return mask

    def _parse(self, fids):
        """ Return the PIDs and CIDs extracted from the FIDs. """
        pids = []
        cids = []
        for fid in fids:
            filename = os.path.splitext(os.path.basename(fid))[0]
            pid, cid = self.regexp.match(filename).groups()
            pids.append(pid)
            cids.append(cid)
        return np.asarray(pids), np.asarray(cids)
