import tensorflow as tf

# Compute the jacobians
def compute_jacobian(model, inputs):
    jacobians = []
    for input in inputs:
        with tf.GradientTape(persistent=True) as tape:
            prediction = model(input[tf.newaxis, ...])
        grads = tape.gradient(prediction, model.trainable_weights)
        flattened_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        jacobians.append(flattened_grads)
        del tape
    return tf.stack(jacobians)

# Compute the tangent kernel as described by Jacot et al. (2018)
# "Neural Tangent Kernel: Convergence and Generalization in Neural Networks": <https://arxiv.org/abs/1806.07572>
def compute_kernel(model, inputs):
    jacobians = compute_jacobian(model, inputs)
    kernel = tf.matmul(jacobians, jacobians, transpose_b=True)
    return kernel

# Function to compute the pairwise interference (cosine similarity) as proposed by Liu (2019)
# "Sparse Representation Neural Networks for Online Reinforcement Learning": <https://era.library.ualberta.ca/items/b4cd1257-69ae-4349-9de6-3feed2648eb1/view/d301ebee-7c64-4027-9411-ed0ef19d6e8f/Liu_Vincent_201909_MSc.pdf>
def pairwise_interference(model, inputs):
    jacobians = compute_jacobian(model, inputs)
    norms = tf.norm(jacobians, axis=1, keepdims=True)
    norm_product = tf.matmul(norms, norms, transpose_b=True)
    cosine_similarity = tf.matmul(jacobians, jacobians, transpose_b=True) / norm_product
    return cosine_similarity

# Compute avg. pairwise interferences for all off-diagonal entries
def avg_pairwise_interference(model, inputs):
    pi_matrix = pairwise_interference(model, inputs)
    # Exclude the diagonal elements when averaging
    mask = 1 - tf.eye(pi_matrix.shape[0])
    masked_pi_matrix = pi_matrix * mask
    # Compute the average of off-diagonal elements (i.e., excluding self-interference terms)
    average_pi = tf.reduce_sum(masked_pi_matrix) / tf.reduce_sum(mask)
    return average_pi

# Compute row-ratios of the Kernel as proposed by Achiam et al. (2020)
# "Towards characterizing divergence in deep q-learning": <https://arxiv.org/abs/1903.08894>
def compute_row_ratios(model, inputs):
    K = compute_kernel(model, inputs)
    N = tf.shape(K)[0]
    mask = 1 - tf.eye(N)
    abs_off_diagonal = tf.abs(K) * mask # Compute the absolute values of off-diagonal elements
    sum_abs_off_diagonal = tf.reduce_sum(abs_off_diagonal, axis=1) # Sum up all the absolute off-diagonal elements in each row
    K_ii = tf.linalg.diag_part(K) # Retrieve diagonal elements
    N_float = tf.cast(N, K_ii.dtype)
    row_ratios = sum_abs_off_diagonal / (N_float * K_ii) # Compute the ratio of the average off-diagonal sum to the diagonal element for each row
    return row_ratios