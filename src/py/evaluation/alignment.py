import gc
import itertools
import multiprocessing
import time
import numpy as np
#from scipy.spatial.distance import cdist

# from src.modules.finding.similarity import sim
from src.py.evaluation.similarity import sim
from src.py.util.util import merge_dic


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate):
    """
        Search alignment with greedy strategy.

        Parameters
        ----------
        embed1 : matrix_like
            An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
        embed2 : matrix_like
            An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
        top_k : list of integers
            Hits@k metrics for evaluating results.
        nums_threads : int
            The number of threads used to search alignment.
        metric : string
            The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
        normalize : bool, true or false.
            Whether to normalize the input embeddings.
        csls_k : int
            K value for csls. If k > 0, enhance the similarity by csls.

        Returns
        -------
        alignment_rest :  list, pairs of aligned entities
        hits1 : float, hits@1 values for alignment results
        mr : float, MR values for alignment results
        mrr : float, MRR values for alignment results
        """
    t = time.time()
    sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls: csls={}, hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(csls_k, top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls: csls={}, hits@{} = {}%, time = {:.3f} s ".format(csls_k, top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
    hits1 = hits[0]
    #del sim_mat
    gc.collect()
    return alignment_rest, hits1, mr, mrr, sim_mat


def find_alignment(sim_mat, sim_th, k):
    """
    Find potential pairs of aligned entities from the similarity matrix.
    The potential pair (x, y) should satisfy: 1) sim(x, y) > sim_th; 2) y is among the nearest-k neighbors of x.

    Parameters
    ----------
    :param sim_mat:
    :param sim_th:
    :param k:
    :return:
    """
    # 找到大于th的行和列的索引对，放在set中
    potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim threshold")
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors = search_nearest_k(sim_mat, k)
    potential_aligned_pairs &= nearest_k_neighbors
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim and nearest k")
    return potential_aligned_pairs


def search_nearest_k(sim_mat, k):
    assert k > 0
    neighbors = set()
    num = sim_mat.shape[0]
    for i in range(num):
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == num * k
    return neighbors


def check_new_alignment(aligned_pairs, context="check alignment"):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))


def filter_sim_mat(mat, threshold, greater=True, equal=False):
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def arg_sort(idx, sim_mat, prefix1, prefix2):
    candidates = dict()
    for i in range(len(idx)):
        x_i = prefix1 + str(idx[i])
        rank = (-sim_mat[i, :]).argsort()
        y_j = [prefix2 + str(r) for r in rank]
        candidates[x_i] = y_j
    return candidates