from .objects import Loss
from .functional import *

SMOOTH = 1e-5


class JaccardLoss(Loss):
    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
    def __call__(self, gt, pr):
        return 1 - iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None
        )


class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return 1 -f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None
        )


class BinaryCELoss(Loss):
    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def __call__(self, gt, pr):
        return binary_crossentropy(gt, pr, **self.submodules)


class CategoricalCELoss(Loss):
    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        return categorical_crossentropy(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            **self.submodules
        )


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_indexes=None):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        return categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_indexes=self.class_indexes
        )

class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, gt, pr):
        return binary_focal_loss(gt, pr, alpha=self.alpha, gamma=self.gamma)


# aliases
jaccard_loss = JaccardLoss()
dice_loss = DiceLoss()

binary_focal_loss = BinaryFocalLoss()
categorical_focal_loss = CategoricalFocalLoss()

binary_crossentropy = BinaryCELoss()
categorical_crossentropy = CategoricalCELoss()

# loss combinations
bce_dice_loss = binary_crossentropy + dice_loss
bce_jaccard_loss = binary_crossentropy + jaccard_loss

cce_dice_loss = categorical_crossentropy + dice_loss
cce_jaccard_loss = categorical_crossentropy + jaccard_loss

binary_focal_dice_loss = binary_focal_loss + dice_loss
binary_focal_jaccard_loss = binary_focal_loss + jaccard_loss

categorical_focal_dice_loss = categorical_focal_loss + dice_loss
categorical_focal_jaccard_loss = categorical_focal_loss + jaccard_loss