import matplotlib.pyplot as plt
import numpy as np


def compute_distances(x, u):
    """
    Compute the distance between each point in X and each point
    in U .

    Input / Output: MATRIX of dists
    """
    dists = np.zeros((x.shape[0], u.shape[0]))
    for ind, ui in enumerate(u):
        dists[:, ind] = np.sqrt(np.square(x[:, 0] - u[ind][0]) + np.square(x[:, 1] - u[ind][1]))
    return dists


def J(x, k):
    sum = 0
    u, w, d = k_means(x, k)
    d = np.square(d)
    for j in range(k):
        w_j_col = w[:, j]
        examples_ind = [i for i, ignr in enumerate(w_j_col) if ignr == 1]  # all of the ones
        for i in examples_ind:
            sum += d[i][j]
    return sum


def k_optimal(x):
    li = []
    for ki in range(1, 11):
        li.append(J(x, ki))

    plt.plot(np.arange(1, 11), li, 'b')
    plt.plot(np.arange(1, 11), li, 'ko')
    plt.xlabel('K')
    plt.ylabel('J() of k')
    plt.show()

    li.reverse()
    threshold = 1700
    for i in range(len(li) - 1):
        if li[i + 1] - li[i] > threshold:
            return len(li) - i  # optimal K


def k_means(x, k):
    U = [(np.random.randint(0, 10), np.random.randint(0, 10))] * k

    iterations = 50
    W = np.zeros((x.shape[0], k))
    dist = None
    for itr in range(iterations):
        W = np.zeros((x.shape[0], k))
        dist = compute_distances(x, np.array(U))
        for i in range(x.shape[0]):
            W[i, np.argmin(dist[i, :])] = 1

        for j in range(k):
            u_j_col = W[:, j]
            examples_ind = [i for i, ignr in enumerate(u_j_col) if ignr == 1]
            leng = len(examples_ind)
            u_x = 0
            u_y = 0
            for e in examples_ind:
                u_x += x[e][0]
                u_y += x[e][1]

            if leng == 0:
                continue
            u_x = float(u_x / leng)
            u_y = float(u_y / leng)
            U[j] = (u_x, u_y)

    return U, W, dist


M = 500
N = 2  # 2 features
# lamda = np.linalg.eig([[1, 0], [0, 3]])
# (3, 1), (0, 6), (2,)
np.random.seed()
x1 = np.random.multivariate_normal([3, 1], np.array([[1, 0], [0, 3]]), M)
x2 = np.random.multivariate_normal([0, 6], np.array([[2, 0], [0, 2]]), M)
x3 = np.random.multivariate_normal([2, 4], np.array([[3, 0], [0, 1]]), M)

print(f'Recover mean of x2: \n{np.mean(x2, axis=0)}\nShould be almost [0, 6]\n')
print(f'Recover cov of x2: \n{np.cov(np.stack((x2[:, 0], x2[:, 1]), axis=0))}\nShould be almost [[2, 0], [0, 2]]\n')

plt.style.use('fivethirtyeight')
plt.title('The 3 different classes we want to cluster')
plt.plot(x1[:, 0], x1[:, 1], 'x')
plt.plot(x2[:, 0], x2[:, 1], 'x')
plt.plot(x3[:, 0], x3[:, 1], 'x')
plt.axis('equal')
plt.show()

X = np.vstack((x1, (np.vstack((x2, x3)))))

K = k_optimal(X)


us, belongs, distance = k_means(X, K)

plt.title(f'K-means with k = {K}')
plt.plot(x1[:, 0], x1[:, 1], 'x')
plt.plot(x2[:, 0], x2[:, 1], 'x')
plt.plot(x3[:, 0], x3[:, 1], 'x')
for ind, ui in enumerate(us):
    plt.plot(ui[0], ui[1], 'ko', label=f'U{ind}')
plt.axis('equal')
plt.legend(loc='best')
plt.show()