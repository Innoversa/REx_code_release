import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch import nn, optim, autograd
from torchmetrics.functional import f1_score
import os, sys

sys.path.append("/home/ugrads/c/clearloveyanzhen/CufflessBP/bioz_processing")
import sicong_util as su
import matplotlib.pyplot as plt
from pprint import pprint

# os.environ['CUDA_VISIBLE_DEVICES'] = "5"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2list(v):
    return v.split(",")


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description="Colored MNIST")
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--n_restarts", type=int, default=1)
parser.add_argument("--penalty_anneal_iters", type=int, default=100)
parser.add_argument("--penalty_weight", type=float, default=10000.0)
parser.add_argument("--steps", type=int, default=101)
parser.add_argument("--grayscale_model", type=str2bool, default=False)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--print_eval_intervals", type=str2bool, default=True)

parser.add_argument("--erm_amount", type=float, default=1.0)

parser.add_argument("--early_loss_mean", type=str2bool, default=True)

parser.add_argument("--rex", type=str2bool, default=True)
parser.add_argument("--mse", type=str2bool, default=True)

parser.add_argument("--plot", type=str2bool, default=True)
parser.add_argument("--save_numpy_log", type=str2bool, default=False)

parser.add_argument("--test_index", type=int, default=2)
parser.add_argument(
    "--npy_data_path",
    default="../../../subject_variance_analysis/batched_data/trial2trial/",
)
parser.add_argument("--wandb_tag", type=str, default="default_tag")
parser.add_argument("--wandb_project", type=str, default="bioz_data_shift")
parser.add_argument("--use_wandb", type=str2bool, default=True)
parser.add_argument("--pred_content", default="binary_130_140")
parser.add_argument("--pnames", type=str, default="75A")
parser.add_argument("--sel_sessions", type=str2list, default="05,06,07")
parser.add_argument("--feature_keyword", default="Morph")


flags = parser.parse_args()

# TODO: logging
all_train_nlls = -1 * np.ones((flags.n_restarts, flags.steps))
all_train_accs = -1 * np.ones((flags.n_restarts, flags.steps))
# all_train_penalties = -1*np.ones((flags.n_restarts, flags.steps))
all_irmv1_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_rex_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_test_accs = -1 * np.ones((flags.n_restarts, flags.steps))
# all_grayscale_test_accs = -1*np.ones((flags.n_restarts, flags.steps))
final_train_accs = []
final_test_accs = []
highest_test_accs = []


def label_binary_converter(
    in_x,
    in_y,
    bp_type="sbp",
    in_vals=[140, 150],
    thre=[-2, +2],
    split_size=0.2,
    out_vals=[0, 1],
):
    # in_x=X; in_y=y; bp_type='sbp'; in_vals=[120, 160]; thre=[-2, +2]; out_vals=[0,1]; balanced_out=True
    if len(in_y.shape) == 2:
        if bp_type == "sbp":
            in_y = in_y[:, 0]
        else:
            in_y = in_y[:, 1]
    if in_vals[0] == 0:
        out_y = in_y[
            ((in_y >= in_vals[1] - 10 + thre[0]) & (in_y <= in_vals[1] + 10 + thre[1]))
        ]
        # plt.plot(out_y, 'r.', alpha=0.3, label='continuous')
        # plt.plot(np.where((out_y >= in_vals[1]+thre[0])&(out_y <= in_vals[1]+thre[1]), in_vals[1], max(in_vals[0], 110)), 'k.', alpha=0.2, label='discrete')
        out_y = np.where(
            (out_y >= in_vals[1] + thre[0]) & (out_y <= in_vals[1] + thre[1]),
            out_vals[1],
            out_vals[0],
        )
        out_x = in_x[
            ((in_y >= in_vals[1] - 10 + thre[0]) & (in_y <= in_vals[1] + 10 + thre[1]))
        ]
    else:
        out_y = in_y[
            ((in_y >= in_vals[0] + thre[0]) & (in_y <= in_vals[0] + thre[1]))
            | ((in_y >= in_vals[1] + thre[0]) & (in_y <= in_vals[1] + thre[1]))
        ]
        # plt.plot(out_y, 'r.', alpha=0.3, label='continuous')
        # plt.plot(np.where((out_y >= in_vals[1]+thre[0])&(out_y <= in_vals[1]+thre[1]), in_vals[1], max(in_vals[0], 110)), 'k.', alpha=0.2, label='discrete')
        out_y = np.where(out_y < in_vals[1] + thre[0], 0, 1)
        out_x = in_x[
            ((in_y >= in_vals[0] + thre[0]) & (in_y <= in_vals[0] + thre[1]))
            | ((in_y >= in_vals[1] + thre[0]) & (in_y <= in_vals[1] + thre[1]))
        ]
    return out_x, out_y


# load and process data
def load_bioz_data():
    bioz_data_list = []
    bioz_label_list = []

    bioz_npy_list = os.listdir(flags.npy_data_path)
    for bioz_npy in bioz_npy_list:
        (
            current_xy,
            current_pname,
            current_feature,
            current_session,
            current_trial,
        ) = bioz_npy[:-4].split()
        if current_feature == flags.feature_keyword:
            if current_session in flags.sel_sessions and current_pname == flags.pnames:
                if "X_file" == current_xy:
                    bioz_data_list.append(flags.npy_data_path + bioz_npy)
                elif "y_file" == current_xy:
                    bioz_label_list.append(flags.npy_data_path + bioz_npy)
    bioz_data_list.sort()
    bioz_label_list.sort()
    print(f"There are {len(bioz_data_list)} selected domains/environments")
    pprint(bioz_data_list)
    pprint(bioz_label_list)
    print("-".join(["-" for i in range(35)]))
    return bioz_data_list, bioz_label_list


# build environment
def build_environment_from_trials(bioz_list, label_list, use_cuda=True):
    train_min_length = 10000
    if len(bioz_list) != len(label_list):
        print("data/label mismatch")
    envs = []
    for i in range(len(bioz_list)):
        # load data
        bioz_x = np.load(bioz_list[i])
        bioz_y = np.load(label_list[i])
        # binarize labels
        _, bp1, bp2 = flags.pred_content.split("_")
        in_bp_pairs = [int(bp1), int(bp2)]
        bioz_x, bioz_y = label_binary_converter(
            bioz_x, bioz_y, in_vals=in_bp_pairs, thre=[-1.5, 1.5]
        )
        # getting length
        # At least getting 3 batches by eliminating the ones without enough data
        if len(bioz_x) >= flags.batch_size * 3:
            train_min_length = min(train_min_length, len(bioz_x))
            # convert to torch
            bioz_x = torch.from_numpy(bioz_x)
            bioz_y = torch.from_numpy(bioz_y.reshape(-1, 1))
            print(bioz_list[i], bioz_x.shape, bioz_y.shape)
            if use_cuda:
                envs.append(
                    {
                        "fname": label_list[i],
                        "images": bioz_x.float().cuda(),
                        "labels": bioz_y.float().cuda(),
                    }
                )
            else:
                envs.append(
                    {
                        "fname": label_list[i],
                        "images": bioz_x.float(),
                        "labels": bioz_y.float(),
                    }
                )
    return envs, train_min_length


envs, train_min_length = build_environment_from_trials(bioz_data_list, bioz_label_list)
num_batches = (train_min_length) // flags.batch_size
print(
    f"# of batches=={num_batches}; batch_size={flags.batch_size}; min_length={train_min_length}"
)
print("-".join(["-" for i in range(35)]))
# initialize wandb
print("line 139 == " + envs[flags.test_index]["fname"].split("/")[-1][6:-4])
if flags.use_wandb:
    import wandb

    wandb.init(
        project=flags.wandb_project,
        reinit=True,
        tags=[flags.wandb_tag, envs[flags.test_index]["fname"].split("/")[-1][6:-4]],
    )
    wandb.config.update(flags)
    log_dict = {}
print("Flags:")
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

for restart in range(flags.n_restarts):
    print("Restart", restart + 1)
    highest_test_acc = 0.0
    # define model
    layer_size = envs[0]["images"].shape[-1]

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(layer_size, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 1)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], layer_size)
            out = self._main(out)
            return out

    if use_cuda:
        mlp = MLP().cuda()
    else:
        mlp = MLP()
    if flags.use_wandb:
        wandb.watch(mlp)
    # define loss functions
    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.0).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def calc_f1_score(logits, y):
        preds = (logits > 0.0).int()
        return f1_score(preds, y.int(), num_classes=2)

    def penalty(logits, y):
        if use_cuda:
            scale = torch.tensor(1.0).cuda().requires_grad_()
        else:
            scale = torch.tensor(1.0).requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode="fixed")
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    def rex_calc(loss_list, flags_test_index, flag_mse):
        rex_pen = 0
        for edx1 in range(len(loss_list[:])):
            for edx2 in range(len(loss_list[edx1:])):
                if (
                    edx1 != edx2
                    and edx1 != flags_test_index
                    and edx2 != flags_test_index
                ):
                    if flag_mse:
                        rex_pen += (
                            loss_list[edx1].mean() - loss_list[edx2].mean()
                        ) ** 2
                    else:
                        rex_pen += (
                            loss_list[edx1].mean() - loss_list[edx2].mean()
                        ).abs()
        return rex_pen

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    pretty_print(
        "step", "train nll", "train acc", "rex penalty", "irmv1 penalty", "test acc"
    )
    i = 0
    for step in range(flags.steps):
        n = i % num_batches
        for edx, env in enumerate(envs):
            if edx != flags.test_index:
                logits = mlp(
                    env["images"][n * flags.batch_size : (n + 1) * flags.batch_size]
                )
                env["nll"] = mean_nll(
                    logits,
                    env["labels"][n * flags.batch_size : (n + 1) * flags.batch_size],
                )
                env["acc"] = calc_f1_score(
                    logits,
                    env["labels"][n * flags.batch_size : (n + 1) * flags.batch_size],
                )
                env["penalty"] = penalty(
                    logits,
                    env["labels"][n * flags.batch_size : (n + 1) * flags.batch_size],
                )
            else:
                logits = mlp(env["images"])
                env["nll"] = mean_nll(logits, env["labels"])
                env["acc"] = calc_f1_score(logits, env["labels"])
                env["penalty"] = penalty(logits, env["labels"])
        i += 1

        train_nll = torch.stack(
            [env["nll"] for edx, env in enumerate(envs) if edx != flags.test_index]
        ).mean()
        train_acc = torch.stack(
            [env["acc"] for edx, env in enumerate(envs) if edx != flags.test_index]
        ).mean()
        irmv1_penalty = torch.stack(
            [env["penalty"] for edx, env in enumerate(envs) if edx != flags.test_index]
        ).mean()

        if use_cuda:
            weight_norm = torch.tensor(0.0).cuda()
        else:
            weight_norm = torch.tensor(0.0)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss_list = [
            env["nll"] for edx, env in enumerate(envs) if edx != flags.test_index
        ]

        if flags.early_loss_mean:
            loss_list = [loss_unit.mean() for loss_unit in loss_list]

        loss = 0.0
        loss += flags.erm_amount * sum(loss_list)

        loss += flags.l2_regularizer_weight * weight_norm

        penalty_weight = (
            flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
        )

        rex_penalty = rex_calc(loss_list, flags.test_index, flags.mse)
        # if flags.mse:
        #     rex_penalty = (loss_list[0].mean() - loss_list[1].mean()) ** 2
        # else:
        #     rex_penalty = (loss_list[0].mean() - loss_list[1].mean()).abs()

        if flags.rex:
            loss += penalty_weight * rex_penalty
        else:
            loss += penalty_weight * irmv1_penalty

        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[flags.test_index]["acc"]
        # grayscale_test_acc = envs[3]['acc']

        if step % flags.eval_interval == 0:
            train_acc_scalar = train_acc.detach().cpu().numpy()
            test_acc_scalar = test_acc.detach().cpu().numpy()
            if flags.print_eval_intervals:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    rex_penalty.detach().cpu().numpy(),
                    irmv1_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy(),
                )
            if (train_acc_scalar >= test_acc_scalar) and (
                test_acc_scalar > highest_test_acc
            ):
                highest_test_acc = test_acc_scalar

        all_train_nlls[restart, step] = train_nll.detach().cpu().numpy()
        all_train_accs[restart, step] = train_acc.detach().cpu().numpy()
        all_rex_penalties[restart, step] = rex_penalty.detach().cpu().numpy()
        all_irmv1_penalties[restart, step] = irmv1_penalty.detach().cpu().numpy()
        all_test_accs[restart, step] = test_acc.detach().cpu().numpy()
        # all_grayscale_test_accs[restart, step] = grayscale_test_acc.detach().cpu().numpy()
    print("highest test acc this run:", highest_test_acc)
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    highest_test_accs.append(highest_test_acc)
    print("Final train acc (mean/std across restarts so far):")
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print("Final test acc (mean/std across restarts so far):")
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print("Highest test acc (mean/std across restarts so far):")
    print(np.mean(highest_test_accs), np.std(highest_test_accs))


plot_x = np.linspace(0, flags.steps, flags.steps)
if flags.plot:
    from pylab import *

    figure()
    xlabel("epoch")
    ylabel("loss")
    title("train/test accuracy")
    plot(plot_x, all_train_accs.mean(0), ls="dotted", label="train_acc")
    plot(plot_x, all_test_accs.mean(0), label="test_acc")
    # plot(plot_x, all_grayscale_test_accs.mean(0), ls="--", label='grayscale_test_acc')
    legend(prop={"size": 11}, loc="upper right")
    savefig("train_acc__test_acc_bioz.pdf")

    figure()
    title("train nll / penalty ")
    plot(plot_x, all_train_nlls.mean(0), ls="dotted", label="train_nll")
    plot(plot_x, all_irmv1_penalties.mean(0), ls="--", label="irmv1_penalty")
    plot(plot_x, all_rex_penalties.mean(0), label="rex_penalty")
    yscale("log")
    legend(prop={"size": 11}, loc="upper right")
    savefig("train_nll__penalty_bioz.pdf")

if flags.save_numpy_log:
    directory = "np_arrays_paper"
    if not os.path.exists(directory):
        os.makedirs(directory)

        outfile = "all_train_nlls"
        np.save(directory + "/" + outfile, all_train_nlls)

        outfile = "all_irmv1_penalties"
        np.save(directory + "/" + outfile, all_irmv1_penalties)

        outfile = "all_rex_penalties"
        np.save(directory + "/" + outfile, all_rex_penalties)

        outfile = "all_train_accs"
        np.save(directory + "/" + outfile, all_train_accs)

        outfile = "all_test_accs"
        np.save(directory + "/" + outfile, all_test_accs)

        # outfile = "all_grayscale_test_accs"
        # np.save(directory + "/" + outfile, all_grayscale_test_accs)

if flags.use_wandb:
    log_dict.update(
        {
            "plot_x": plot_x,
            "all_train_nlls": all_train_nlls.mean(0),
            "all_irmv1_penalties": all_irmv1_penalties.mean(0),
            "all_train_accs": all_train_accs.mean(0),
            "all_test_accs": all_test_accs.mean(0),
            "final_train_accs": all_train_accs.mean(0)[-1],
            "final_test_accs": all_test_accs.mean(0)[-1],
        }
    )
    wandb.log(log_dict)
    wandb.join()
    # print(log_dict)
