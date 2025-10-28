import torch
from torch import nn
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from contextlib import suppress
from timm.utils import NativeScaler
try:
    from apex import amp
    from timm.utils import ApexScaler
    has_apex = True
except ImportError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


def loss_adv(loss_name, outputs, labels, target_labels, target, device):
    '''The function to create loss function.'''
    if loss_name=="ce":
        loss = nn.CrossEntropyLoss()
        
        if target:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

    elif loss_name =='cw':
        if target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
            cost = cost.sum()
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
            cost = cost.sum()
    return cost

def margin_loss(outputs, labels, target_labels, targeted, device):
    '''Define the margin loss.'''
    if targeted:
        one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
    else:
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
    return cost.sum()

def resolve_amp(args, _logger):
    '''The function to resolve amp parameters for robust training.'''
    args.amp_version=''
    # resolve AMP arguments based on PyTorch / Apex availability
    if args.apex_amp and has_apex:
        args.amp_version = 'apex'
    elif args.native_amp and has_native_amp:
        args.amp_version = 'native'
    else:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")


def build_loss_scaler(args, _logger):
    '''The function to build loss scaler for robust training.'''
    # setup loss scaler
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp_version == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif args.amp_version == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        _logger.info('AMP not enabled. Training in float32.')
        
    return amp_autocast, loss_scaler


def build_loss(args, mixup_fn, num_aug_splits):
    '''The function to build loss function for robust training.'''
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_fn is not None:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #  Case 2: DBP (GradNorm) loss
    elif args.gradnorm:
        train_loss_fn = GradNorm_Loss(
            eps=args.attack_eps,
            lambda_ce=getattr(args, "lambda_ce", 1.0),
            lambda_gn=getattr(args, "lambda_gn", 1.0),
        )
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    return train_loss_fn, validate_loss_fn



class GradNorm_Loss(nn.Module):
    """
    Double Backpropagation Loss (GradNorm-style).
    Implements:
        L = λ_CE * CE(fθ(x), y) + λ_GN * (ε / σ) * ||∇ₓ CE(fθ(x), y)||₁
    """

    def __init__(self, eps=4./255., std=0.225, lambda_ce=0.8, lambda_gn=1.2):
        super().__init__()
        self.eps = eps / std              # ε / σ scaling
        self.lambda_ce = lambda_ce
        self.lambda_gn = lambda_gn
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, model, x, y):
        # Enable gradients w.r.t. input
        x.requires_grad_(True)

        # Forward pass and CE loss
        logits = model(x)
        loss_ce = self.cross_entropy(logits, y)

        # Gradient w.r.t input
        grad = torch.autograd.grad(loss_ce, x, create_graph=True)[0]

        # GradNorm penalty (L1 norm)
        grad_norm = grad.abs().sum(dim=(1, 2, 3)).sum()

        # Total loss
        loss = (self.lambda_ce * loss_ce
                + self.lambda_gn * self.eps * grad_norm)
        return loss