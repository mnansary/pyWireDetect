import tensorflow as tf
SMOOTH = 1e-5


# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------

def _gather_channels(x, indexes):
    x = tf.keras.backend.permute_dimensions(x, (3, 0, 1, 2))
    x = tf.keras.backend.gather(x, indexes)
    x = tf.keras.backend.permute_dimensions(x, (1, 2, 3, 0))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] 
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = tf.keras.backend.greater(x, threshold)
        x = tf.keras.backend.cast(x, tf.keras.backend.floatx())
    return x


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = tf.keras.backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return tf.keras.backend.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1.,
                 class_indexes=None, 
                 smooth=SMOOTH, 
                 per_image=False, 
                 threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    # score calculation
    intersection = tf.keras.backend.sum(gt * pr, axis=axes)
    union = tf.keras.backend.sum(gt + pr, axis=axes) - intersection
    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights)
    return score


def f_score(gt, pr, beta=1, 
            class_weights=1, 
            class_indexes=None, 
            smooth=SMOOTH,
            per_image=False, 
            threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # calculate score
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    fn = tf.keras.backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights)
    return score


def precision(gt, pr, class_weights=1, 
                class_indexes=None, 
                smooth=SMOOTH, 
                per_image=False, 
                threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    # score calculation
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights)
    return score


def recall(gt, pr, class_weights=1, 
                class_indexes=None, 
                smooth=SMOOTH, 
                per_image=False, 
                threshold=None):
    
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fn = tf.keras.backend.sum(gt, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights)
    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., 
                            class_indexes=None):

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    axis = 3 
    pr /= tf.keras.backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # calculate loss
    output = gt * tf.keras.backend.log(pr) * class_weights
    return - tf.keras.backend.mean(output)


def binary_crossentropy(gt, pr):
    return tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, 
                            alpha=0.25, class_indexes=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    # Calculate focal loss
    loss = - gt * (alpha * tf.keras.backend.pow((1 - pr), gamma) * tf.keras.backend.log(pr))
    return tf.keras.backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    # clip to prevent NaN's and Inf's
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    loss_1 = - gt * (alpha * tf.keras.backend.pow((1 - pr), gamma) * tf.keras.backend.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * tf.keras.backend.pow((pr), gamma) * tf.keras.backend.log(1 - pr))
    loss = tf.keras.backend.mean(loss_0 + loss_1)
    return loss