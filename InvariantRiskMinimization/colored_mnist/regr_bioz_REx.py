import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch import nn, optim, autograd
from torchmetrics.functional import f1_score
from scipy.stats import pearsonr
import os, sys

sys.path.append("/home/ugrads/c/clearloveyanzhen/CufflessBP/bioz_processing")
import sicong_util as su
import matplotlib.pyplot as plt
from pprint import pprint

global flags, use_cuda, wandb
import math

use_cuda = torch.cuda.is_available()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2list(v):
    return v.split(",")


def parsing_args():
    parser = argparse.ArgumentParser(description="bioz_rex_util")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--l2_regularizer_weight", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_restarts", type=int, default=1)
    parser.add_argument("--penalty_anneal_iters", type=int, default=100)
    parser.add_argument("--penalty_weight", type=float, default=10000.0)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=0.1)
    parser.add_argument("--print_eval_intervals", type=str2bool, default=True)
    parser.add_argument("--erm_amount", type=float, default=1.0)
    parser.add_argument("--early_loss_mean", type=str2bool, default=True)
    parser.add_argument("--rex", type=str2bool, default=True)
    parser.add_argument("--mse", type=str2bool, default=True)
    parser.add_argument(
        "--npy_data_path",
        default="../../../subject_variance_analysis/batched_data/trial2trial/",
    )

    parser.add_argument("--test_index", type=int, default=2)
    parser.add_argument("--wandb_tag", type=str, default="default_tag")
    parser.add_argument("--wandb_project", type=str, default="bioz_data_shift_regr")
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--pred_content", default="regr_130_140")
    parser.add_argument("--pnames", type=str, default="75A")
    parser.add_argument("--sel_sessions", type=str2list, default="05,06,07")
    parser.add_argument("--feature_keyword", default="Morph")
    parser.add_argument("--f", default="default")
    parser.add_argument("--jupyter_notebook", type=str2bool, default=False)
    parser.add_argument("--gpu_core", default="5")
    parser.add_argument("--normalize_x", type=str2bool, default=True)

    flags = parser.parse_args()
    flags.eval_interval = min(100, max(50, int(flags.eval_interval * flags.steps)))
    return flags


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


def label_regr_converter(in_x, in_y, bp_type="sbp"):
    if len(in_y.shape) == 2:
        if bp_type == "sbp":
            in_y = in_y[:, 0]
        else:
            in_y = in_y[:, 1]
    return in_x, in_y


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
    # pprint(bioz_data_list)
    # pprint(bioz_label_list)
    print("-".join(["-" for i in range(35)]))
    return bioz_data_list, bioz_label_list


# Find Min-Max values of all the files
def find_min_max(label_list):
    min_val = math.inf
    max_val = -math.inf
    for lf in label_list:
        lbw = np.load(lf)
        min_val = min(min_val, lbw.min())
        max_val = max(max_val, lbw.max())
    print(f"min={min_val}; max={max_val}")
    return min_val, max_val


# build environment
def build_environment_from_trials(bioz_list, label_list):
    train_min_length = 10000
    if flags.normalize_x:
        xmin_val, xmax_val = find_min_max(bioz_list)
        flags.x_norm = su.Sicong_Norm(min_val=xmin_val, max_val=xmax_val)
        # ymin_val, ymax_val = find_min_max(label_list)
        # flags.y_norm = su.Sicong_Norm(min_val=ymin_val, max_val=ymax_val)
    if len(bioz_list) != len(label_list):
        print("data/label mismatch")
    envs = []

    for i in range(len(bioz_list)):
        # load data
        bioz_x = np.load(bioz_list[i])
        bioz_y = np.load(label_list[i])
        # binarize labels
        goal, bp1, bp2 = flags.pred_content.split("_")
        in_bp_pairs = [int(bp1), int(bp2)]
        if goal == "binary":
            bioz_x, bioz_y = label_binary_converter(
                bioz_x, bioz_y, in_vals=in_bp_pairs, thre=[-1.5, 1.5]
            )
        elif goal == "regr":
            bioz_x, bioz_y = label_regr_converter(bioz_x, bioz_y)
        # getting length
        # At least getting 3 batches by eliminating the ones without enough data
        if len(bioz_x) >= flags.batch_size * 3:
            train_min_length = min(train_min_length, len(bioz_x))
            # convert to torch
            bioz_x = torch.from_numpy(bioz_x)
            bioz_y = torch.from_numpy(bioz_y.reshape(-1, 1))
            bioz_fname = bioz_list[i].split("/")[-1][7:-4]
            print(bioz_fname, bioz_x.shape, bioz_y.shape)
            # bioz_ny = flags.y_norm.normalize(bioz_y)
            if flags.normalize_x:
                bioz_x = flags.x_norm.normalize(bioz_x)
            if use_cuda:
                envs.append(
                    {
                        "fname": bioz_fname,
                        "images": bioz_x.float().cuda(),
                        "labels": bioz_y.float().cuda(),
                    }
                )
            else:
                envs.append(
                    {
                        "fname": bioz_fname,
                        "images": bioz_x.float(),
                        "labels": bioz_y.float(),
                    }
                )
    return envs, train_min_length


# define model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        lin1 = nn.Linear(flags.layer_size, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input.view(input.shape[0], flags.layer_size)
        out = self._main(out)
        return out


# define loss functions
def mean_nll(logits, y):
    # return F.binary_cross_entropy_with_logits(logits, y)
    return F.mse_loss(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def calc_f1_score(logits, y):
    preds = (logits > 0.0).int()
    return f1_score(preds, y.int(), num_classes=2)


def calc_RMSE(logits, y):
    return torch.sqrt(F.mse_loss(logits, y))


def calc_Pearson(logits, y):
    r_val, p_val = pearsonr(
        logits.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy().reshape(-1)
    )
    return torch.tensor(r_val).float().cuda()


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
            if edx1 != edx2 and edx1 != flags_test_index and edx2 != flags_test_index:
                if flag_mse:
                    rex_pen += (loss_list[edx1].mean() - loss_list[edx2].mean()) ** 2
                else:
                    rex_pen += (loss_list[edx1].mean() - loss_list[edx2].mean()).abs()
    return rex_pen


def fit_model(envs):
    all_train_nlls = -1 * np.ones((flags.n_restarts, flags.steps))
    all_train_accs = -1 * np.ones((flags.n_restarts, flags.steps))
    all_train_pearsons = -1 * np.ones((flags.n_restarts, flags.steps))
    # all_train_penalties = -1*np.ones((flags.n_restarts, flags.steps))
    all_irmv1_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
    all_rex_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
    all_test_accs = -1 * np.ones((flags.n_restarts, flags.steps))
    all_test_pearsons = -1 * np.ones((flags.n_restarts, flags.steps))
    final_train_accs = []
    final_train_pearsons = []
    final_test_accs = []
    final_test_pearsons = []
    best_test_accs = []
    best_model = None
    best_loss = 0.0
    # Swapping Notebook Progress Bar
    if flags.jupyter_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    for restart in range(flags.n_restarts):
        best_test_acc = 0.0
        if use_cuda:
            mlp = MLP().cuda()
        else:
            mlp = MLP()
        if flags.use_wandb:
            wandb.watch(mlp)
        optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
        print("Restart", restart + 1)
        i = 0
        steps_enumerator = np.arange(flags.steps)
        if flags.print_eval_intervals:
            pretty_print(
                "step",
                "train nll",
                "train acc",
                "train_pearson",
                "rex penalty",
                "irmv1 penalty",
                "test acc",
                "test_pearson",
            )
        else:
            steps_enumerator = tqdm(steps_enumerator)
        for step in steps_enumerator:
            n = i % flags.num_batches
            for edx, env in enumerate(envs):
                if edx != flags.test_index:
                    logits = mlp(
                        env["images"][n * flags.batch_size : (n + 1) * flags.batch_size]
                    )
                    env["nll"] = mean_nll(
                        logits,
                        env["labels"][
                            n * flags.batch_size : (n + 1) * flags.batch_size
                        ],
                    )
                    env["acc"] = calc_RMSE(
                        logits,
                        env["labels"][
                            n * flags.batch_size : (n + 1) * flags.batch_size
                        ],
                    )
                    env["penalty"] = penalty(
                        logits,
                        env["labels"][
                            n * flags.batch_size : (n + 1) * flags.batch_size
                        ],
                    )
                    env["pearson_r"] = calc_Pearson(
                        logits,
                        env["labels"][
                            n * flags.batch_size : (n + 1) * flags.batch_size
                        ],
                    )
                else:
                    logits = mlp(env["images"])
                    env["nll"] = mean_nll(logits, env["labels"])
                    env["acc"] = calc_RMSE(logits, env["labels"])
                    env["penalty"] = penalty(logits, env["labels"])
                    env["pearson_r"] = calc_Pearson(logits, env["labels"])
            i += 1
            train_nll = torch.stack(
                [env["nll"] for edx, env in enumerate(envs) if edx != flags.test_index]
            ).mean()
            train_acc = torch.stack(
                [env["acc"] for edx, env in enumerate(envs) if edx != flags.test_index]
            ).mean()
            irmv1_penalty = torch.stack(
                [
                    env["penalty"]
                    for edx, env in enumerate(envs)
                    if edx != flags.test_index
                ]
            ).mean()
            train_pearson = torch.stack(
                [
                    env["pearson_r"]
                    for edx, env in enumerate(envs)
                    if edx != flags.test_index
                ]
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
            test_pearson = envs[flags.test_index]["pearson_r"]

            if step % flags.eval_interval == 0:
                train_acc_scalar = train_acc.detach().cpu().numpy()
                test_acc_scalar = test_acc.detach().cpu().numpy()
                train_pearson_scalar = train_pearson.detach().cpu().numpy()
                test_pearson_scalar = test_pearson.detach().cpu().numpy()
                if flags.print_eval_intervals:
                    pretty_print(
                        np.int32(step),
                        train_nll.detach().cpu().numpy(),
                        train_acc.detach().cpu().numpy(),
                        train_pearson.detach().cpu().numpy(),
                        rex_penalty.detach().cpu().numpy(),
                        irmv1_penalty.detach().cpu().numpy(),
                        test_acc.detach().cpu().numpy(),
                        test_pearson.detach().cpu().numpy(),
                    )
                if (train_acc_scalar <= test_acc_scalar) and (
                    test_acc_scalar < best_test_acc
                ):
                    best_test_acc = test_acc_scalar

            all_train_nlls[restart, step] = train_nll.detach().cpu().numpy()
            all_train_accs[restart, step] = train_acc.detach().cpu().numpy()
            all_train_pearsons[restart, step] = train_pearson.detach().cpu().numpy()
            all_rex_penalties[restart, step] = rex_penalty.detach().cpu().numpy()
            all_irmv1_penalties[restart, step] = irmv1_penalty.detach().cpu().numpy()
            all_test_accs[restart, step] = test_acc.detach().cpu().numpy()
            all_test_pearsons[restart, step] = test_pearson.detach().cpu().numpy()
        final_train_accs.append(train_acc.detach().cpu().numpy())
        final_test_accs.append(test_acc.detach().cpu().numpy())
        final_train_pearsons.append(train_pearson.detach().cpu().numpy())
        final_test_pearsons.append(test_pearson.detach().cpu().numpy())
        best_test_accs.append(best_test_acc)
        if __name__ == "__main__":
            print("best test acc this run:", best_test_acc)
            print("Final train acc (mean/std across restarts so far):")
            print(np.mean(final_train_accs), np.std(final_train_accs))
            print("Final test acc (mean/std across restarts so far):")
            print(np.mean(final_test_accs), np.std(final_test_accs))
            print("best test acc (mean/std across restarts so far):")
            print(np.mean(best_test_accs), np.std(best_test_accs))
        if final_test_accs[-1] >= best_loss:
            best_model = mlp
            best_loss = final_test_accs[-1]
    print(f"best model with accuracy={best_loss} (end of function)")
    return (
        best_model,
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    )


def plot_progress(
    all_train_accs,
    all_train_pearsons,
    all_test_accs,
    all_test_pearsons,
    all_train_nlls,
    all_irmv1_penalties,
    all_rex_penalties,
):
    plot_x = np.linspace(0, flags.steps, flags.steps)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
    plt.close()
    # train_test_plot
    ax[0].set_title("BioZ_REx_Train_Test_Accuracy")
    ax[0].set_ylabel("loss(f1_score)")
    ax[0].plot(plot_x, all_train_accs.mean(0), ls="-.", label="train_acc")
    ax[0].plot(plot_x, all_test_accs.mean(0), label="test_acc")
    ax[0].legend(loc="upper right")
    # penalty plot_x
    ax[1].set_ylabel("train nll/penalty")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch")
    ax[1].plot(plot_x, all_train_nlls.mean(0), ls="-.", label="train_nll")
    ax[1].plot(plot_x, all_irmv1_penalties.mean(0), ls="--", label="irmv1_penalty")
    ax[1].plot(plot_x, all_rex_penalties.mean(0), label="rex_penalty")
    ax[1].legend(loc="upper right")
    return fig


def print_flags():
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))


def preprocess():
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    bioz_data_list, bioz_label_list = load_bioz_data()
    envs, train_min_length = build_environment_from_trials(
        bioz_data_list, bioz_label_list
    )
    flags.num_batches = train_min_length // flags.batch_size
    flags.layer_size = envs[0]["images"].shape[-1]
    return envs, train_min_length


def main():
    global flags, use_cuda, wandb
    flags = parsing_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_core
    envs, train_min_length = preprocess()
    if flags.use_wandb:
        import wandb

        wandb.init(
            project=flags.wandb_project,
            reinit=True,
            tags=[flags.wandb_tag, envs[flags.test_index]["fname"]],
        )
        wandb.config.update(flags)
        log_dict = {}
    print(
        f"# of batches=={flags.num_batches}; batch_size={flags.batch_size}; min_length={train_min_length}"
    )
    print("-".join(["-" for i in range(35)]))
    print("test_file == " + envs[flags.test_index]["fname"])
    print("Flags:")
    (
        best_mlp,
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    ) = fit_model(envs)
    fig = plot_progress(
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    )
    sicong_logits = best_mlp(envs[flags.test_index]["images"])
    sicong_rmse = calc_RMSE(sicong_logits, envs[flags.test_index]["labels"])
    sicong_pearson = (
        calc_Pearson(sicong_logits, envs[flags.test_index]["labels"])
        .detach()
        .cpu()
        .numpy()
    )
    # print(sicong_logits.detach().cpu().numpy().reshape(-1,1))
    # print(envs[flags.test_index]['labels'].detach().cpu().numpy().reshape(-1,1))
    waveform_visual = su.visual_pred_test(
        pred_arr=sicong_logits.detach().cpu().numpy(),
        test_arr=envs[flags.test_index]["labels"].detach().cpu().numpy(),
        x_lab="time (sec)",
        y_lab="SBP (mmHg)",
        title=f"BioZ_REx_regr for subject {flags.pnames} with sessions {flags.sel_sessions}\n RMSE=={sicong_rmse:.4f} and Pearson=={sicong_pearson:.4f}",
    )
    waveform_visual.savefig("hsc.png")
    if flags.use_wandb:
        log_dict = {
            "all_train_nlls": all_train_nlls.mean(0),
            "waveform": wandb.Image(waveform_visual, caption="final_waveform_sbp"),
            "all_irmv1_penalties": all_irmv1_penalties.mean(0),
            "all_train_accs": all_train_accs.mean(0),
            "all_train_pearsons": all_train_pearsons.mean(0),
            "all_test_accs": all_test_accs.mean(0),
            "all_test_pearsons": all_test_pearsons.mean(0),
            "final_train_accs": all_train_accs.mean(0)[-1],
            "final_train_pearsons": all_train_pearsons.mean(0)[-1],
            "final_test_accs": all_test_accs.mean(0)[-1],
            "final_test_pearsons": all_test_pearsons.mean(0)[-1],
            "plot": fig,
        }
        wandb.log(log_dict)
        wandb.join()


if __name__ == "__main__":
    main()
