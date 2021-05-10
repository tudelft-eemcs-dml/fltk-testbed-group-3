import numpy as np
from queue import Queue

import torch


def random_selection(clients, n):
    return np.random.choice(clients, n, replace=False)


def balanced_sampling(groups, round):
    return groups[round % len(groups)]


# TODO: generate balanced group properly......
def init_groups(size, frac_workers, cls_freq, clients):
    all_groups = []
    all_groups_np = []
    done = False
    gp_size = int(frac_workers * size)

    # 2D array that records if class i exists at worker j or not
    wrk_cls = [[False for i in range(size)] for j in range(size)]
    cls_q = [Queue(maxsize=size) for _ in range(10)]

    for i, cls_list in enumerate(cls_freq):
        wrk_cls[i] = [True if freq != 0 else False for freq in cls_list]

    for worker, class_list in enumerate(reversed(wrk_cls)):
        for cls, exist in enumerate(class_list):
            if exist:
                cls_q[cls].put(size - worker-1)

        # This array counts the number of samples (per class) taken for training so far.
        # The algorithm will try to make the numbers in this array as equal as possible
        taken_count = [0 for i in range(10)]
    while not done:
        # makes sure that we take any worker only once in the group
        visited = [False for i in range(size)]
        g = []
        for _ in range(gp_size):
            # Choose class (that is minimum represented so far)...using "taken_count" array
            cls = np.where(taken_count == np.min(taken_count))[0][0]
            assert cls >= 0 and cls <= len(taken_count)
            # Choose a worker to represnt that class...using wrk_cls and visited array
            done_q = False
            count = 0
            while not done_q:
                wrkr = cls_q[cls].get()
                assert wrk_cls[wrkr][cls]
                if not visited[wrkr] and wrk_cls[wrkr][cls]:
                    # Update the state: taken_count and visited
                    g.append(wrkr)
                    taken_count += cls_freq[wrkr]
                    visited[wrkr] = True
                    done_q = True
                cls_q[cls].put(wrkr)
                count += 1
                # Such an optimal assignment does not exist
                if count == size:
                    done_q = True

        g.append(0)
        g = torch.FloatTensor(g)

        # dist.broadcast(g, src=0)
        g = g.cpu().numpy().tolist()

        # Make sure there is at most one occurrence of "0" in the list of group members
        if g.count(0) > 1:
            g.remove(0)
        # try:
        #     group = dist.new_group(g, timeout=datetime.timedelta(0, timeout))
        # except Exception as e:
        #     done = True
        all_groups_np.append(np.sort(g))
        all_groups.append(group)
        if len(all_groups) > 100:
            done = True

    return all_groups
