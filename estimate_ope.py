from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from obp.dataset import SyntheticBanditDataset, logistic_reward_function, linear_behavior_policy
from obp.policy import IPWLearner
from obp.ope import OffPolicyEvaluation, RegressionModel
from obp.ope import InverseProbabilityWeighting as IPW
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPW
from obp.ope import SwitchDoublyRobust as SwitchDR
from obp.ope import DoublyRobustTuning, SelfNormalizedDoublyRobust, SwitchDoublyRobustTuning, DoublyRobustWithShrinkageTuning

import numpy as np
from obp.utils import softmax
from obp.utils import sample_action_fast

from typing import Optional


class CustomContextBanditDataset(SyntheticBanditDataset):
    def obtain_batch_bandit_feedback(self, n_rounds: int):
        # カスタム特徴量の生成
        age = np.random.randint(20, 81, size=n_rounds)
        homeown = np.random.binomial(1, 0.6, size=n_rounds)

        # 特徴量を結合して2次元配列に変換
        contexts = np.column_stack([age, homeown])
        # 元のメソッドの処理を参考に実装
        # 期待報酬の計算
        expected_reward_ = self.calc_expected_reward(contexts)

        # 行動方策の計算
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        # 行動方策の確率計算
        pi_b = softmax(self.beta * pi_b_logits)

        # アクションのサンプリング
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # 報酬のサンプリング
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        # バンディットフィードバックの作成
        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=expected_reward_,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )

# def custom_behavior_policy(
#     context: np.ndarray,
#     action_context: np.ndarray,
#     random_state: Optional[int] = None,
# ) -> np.ndarray:
#     """年齢と持ち家情報に基づくカスタム行動方策関数

#     Parameters
#     ----------
#     context: array-like, shape (n_rounds, dim_context)
#         コンテキスト（年齢と持ち家情報）

#     action_context: array-like, shape (n_actions, dim_action_context)
#         アクションの特徴表現

#     random_state: int, default=None
#         乱数生成のシード値

#     Returns
#     -------
#     logits: array-like, shape (n_rounds, n_actions)
#         各コンテキスト-アクションペアの行動方策のロジット値
#     """
#     from obp.utils import check_array
#     from sklearn.utils import check_random_state

#     check_array(array=context, name="context", expected_dim=2)
#     check_array(array=action_context, name="action_context", expected_dim=2)

#     random_ = check_random_state(random_state)

#     n_rounds = context.shape[0]
#     n_actions = action_context.shape[0]

#     # 結果を格納する配列
#     logits = np.zeros((n_rounds, n_actions))

#     for i in range(n_rounds):
#         # 年齢（1列目）と持ち家情報（2列目）を取得
#         age = context[i, 0]
#         homeown = context[i, 1]

#         # 傾向スコアの計算
#         logit_ps = -8 + 0.1 * age + 1.0 * homeown
#         ps = 1 / (1 + np.exp(-logit_ps))

#         # アクション0（クーポンを発行しない）の確率は (1-ps)
#         # アクション1（クーポンを発行する）の確率は ps
#         if n_actions == 2:
#             # ロジット値に変換（softmax関数で確率に戻せるように）
#             logits[i, 0] = np.log(1 - ps + 1e-8)  # 数値安定性のため小さな値を加える
#             logits[i, 1] = np.log(ps + 1e-8)
#         else:
#             # n_actionsが2以外の場合は、最初のアクションにpsを割り当て、残りは均等に分配
#             logits[i, 0] = np.log(ps + 1e-8)
#             remaining_prob = (1 - ps) / (n_actions - 1) if n_actions > 1 else 0
#             for j in range(1, n_actions):
#                 logits[i, j] = np.log(remaining_prob + 1e-8)

#     return logits

from typing import Optional, Tuple
import numpy as np

def custom_behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Calculate logits based on age and homeown features"""
    from obp.utils import check_array
    from sklearn.utils import check_random_state

    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # Calculate logits as before
    logits = np.zeros((n_rounds, n_actions))

    for i in range(n_rounds):
        age = context[i, 0]
        homeown = context[i, 1]

        # Calculate propensity score logits
        logit_ps = -8 + 0.1 * age + 1.0 * homeown
        ps = 1 / (1 + np.exp(-logit_ps))

        # Convert to logits for the behavior policy
        if n_actions == 2:
            logits[i, 0] = np.log(1 - ps + 1e-8)
            logits[i, 1] = np.log(ps + 1e-8)
        else:
            logits[i, 0] = np.log(ps + 1e-8)
            remaining_prob = (1 - ps) / (n_actions - 1) if n_actions > 1 else 0
            for j in range(1, n_actions):
                logits[i, j] = np.log(remaining_prob + 1e-8)

    return logits


class TopNTreatmentBanditDataset(SyntheticBanditDataset):
    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        n_treated: Optional[int] = None,
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        n_actions: int
            Number of actions (usually 2 for treatment/control)

        dim_context: int
            Dimension of context vectors

        n_treated: Optional[int]
            Number of users to assign treatment to (top N by propensity score)
            If None, uses standard probabilistic assignment
        """
        super().__init__(n_actions=n_actions, dim_context=dim_context, *args, **kwargs)
        self.n_treated = n_treated

    def obtain_batch_bandit_feedback(self, n_rounds: int):
        # Generate custom features
        age = np.random.randint(20, 81, size=n_rounds)
        homeown = np.random.binomial(1, 0.6, size=n_rounds)

        # Combine features
        contexts = np.column_stack([age, homeown])

        # Calculate expected rewards
        expected_reward_ = self.calc_expected_reward(contexts)

        # Calculate behavior policy logits
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        # Calculate propensity scores
        pi_b = softmax(self.beta * pi_b_logits)

        # Extract propensity scores for action 1 (treatment)
        ps = pi_b[:, 1] if pi_b.shape[1] > 1 else pi_b[:, 0]

        # # Assign treatment based on top-N or probabilistic approach
        # if self.n_treated is None:
        #     # Standard probabilistic assignment
        #     actions = sample_action_fast(pi_b, random_state=self.random_state)
        # else:
        #     # Top-N treatment assignment
        #     # Get indices of top N users by propensity score
        #     treated_idx = np.argsort(ps)[-self.n_treated:]

        #     # Initialize all to control (action 0)
        #     actions = np.zeros(n_rounds, dtype=int)

        #     # Assign treatment (action 1) to top N
        #     actions[treated_idx] = 1

        # # Sample rewards based on actions
        # rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        # # For accurate evaluation, we need to adjust the pscore
        # # to reflect the deterministic assignment
        # if self.n_treated is not None:
        #     # For top-N assignment, pscore is 1.0 for treated, 0.0 for control
        #     pscore = np.zeros(n_rounds)
        #     pscore[treated_idx] = 1.0
        # else:
        #     # Standard probabilistic pscore
        #     pscore = pi_b[np.arange(n_rounds), actions]

        # # Return bandit feedback
        # return dict(
        #     n_rounds=n_rounds,
        #     n_actions=self.n_actions,
        #     context=contexts,
        #     action_context=self.action_context,
        #     action=actions,
        #     position=None,
        #     reward=rewards,
        #     expected_reward=expected_reward_,
        #     pi_b=pi_b[:, :, np.newaxis],j
        #     pscore=pscore,
        # )

        # Assign treatment based on top-N or probabilistic approach
        if self.n_treated is None:
            # Standard probabilistic assignment
            actions = sample_action_fast(pi_b, random_state=self.random_state)
            # Standard probabilistic pscore
            pscore = pi_b[np.arange(n_rounds), actions]
        else:
            # Top-N treatment assignment
            # Get indices of top N users by propensity score
            treated_idx = np.argsort(ps)[-self.n_treated:]

            # Initialize all to control (action 0)
            actions = np.zeros(n_rounds, dtype=int)

            # Assign treatment (action 1) to top N
            actions[treated_idx] = 1

            ## For accurate evaluation, ensure pscores are positive
            #epsilon = 1e-6  # Small positive value
            #pscore = np.ones(n_rounds) * epsilon
            pscore = pi_b[np.arange(n_rounds), actions]
            #pscore[treated_idx] = 1.0

        # Sample rewards based on actions
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        # Return bandit feedback
        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=expected_reward_,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pscore,
        )

import numpy as np
from typing import Optional, Dict
from obp.dataset import SyntheticBanditDataset
from obp.utils import softmax, sample_action_fast

class GroupBasedBanditDataset(SyntheticBanditDataset):
    def __init__(
        self,
        n_actions: int = 2,
        dim_context: int = 2,
        seed: int = 12345,
        *args,
        **kwargs
    ):
        """
        特徴量に基づいてユーザーをグループ分けし、グループごとに異なる反応確率を持つ合成データセット

        Parameters
        ----------
        n_actions: int
            アクション数（通常は処置あり/なしの2値）

        dim_context: int
            コンテキスト特徴量の次元数

        seed: int
            乱数シード
        """
        super().__init__(n_actions=n_actions, dim_context=dim_context, random_state=seed, *args, **kwargs)
        self.seed = seed

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> Dict:
        """バンディットフィードバックデータを生成する

        Parameters
        ----------
        n_rounds: int
            データサイズ

        Returns
        -------
        bandit_feedback: Dict
            バンディットフィードバックデータ
        """
        # 1. 特徴量の生成
        np.random.seed(self.seed)
        N = n_rounds
        age = np.random.randint(20, 81, size=N)
        homeown = np.random.binomial(1, 0.6, size=N)

        # コンテキスト特徴量を結合
        contexts = np.column_stack([age, homeown])

        # 2. 特徴量に基づいたグループ分け
        # グループを決定するための追加特徴量を生成
        np.random.seed(self.seed + 10)
        marketing_receptivity = 0.01 * age + 0.5 * homeown + np.random.normal(0, 1, size=N)
        price_sensitivity = -0.01 * age + 0.3 * (1 - homeown) + np.random.normal(0, 1, size=N)

        # 反応グループを初期化
        response_group = np.zeros(N, dtype=int)

        # 特徴量に基づいてグループを割り当て
        # グループ1: 高年齢 + 高マーケティング受容性 -> 常に成約
        group1_mask = (age > 60) & (marketing_receptivity > 1.5)
        response_group[group1_mask] = 1

        # グループ2: 中年齢 + マーケティング受容性中以上 -> 処置で成約
        group2_mask = (age > 30) & (age <= 60) & (marketing_receptivity > 0) & ~group1_mask
        response_group[group2_mask] = 2

        # グループ3: 中～高年齢 + 高価格感度 -> 処置で離脱
        group3_mask = (age > 40) & (price_sensitivity > 1.0) & ~group1_mask & ~group2_mask
        response_group[group3_mask] = 3

        # グループ4: その他すべて -> 常に不成約
        response_group[(response_group == 0)] = 4

        # 3. 行動方策の計算（傾向スコア）
        # 年齢と持ち家状況に基づいて傾向スコアを計算
        logit_ps = -8 + 0.1 * age + 1.0 * homeown
        ps = 1 / (1 + np.exp(-logit_ps))

        # 4. 処置割り当て
        np.random.seed(self.seed)
        treatment = np.random.binomial(1, ps, size=N)

        # 5. グループごとの確率モデルパラメータ
        # 各グループに対して別々のパラメータでbase_probとtreat_probを計算
        base_prob = np.zeros(N)
        treat_prob = np.zeros(N)

        # グループ1: 常にoutcome=1 (y0=1, y1=1)
        g1_indices = np.where(response_group == 1)[0]
        if len(g1_indices) > 0:
            g1_logit_base = -2 + 0.08 * age[g1_indices] + 0.7 * homeown[g1_indices]
            base_prob[g1_indices] = 1 / (1 + np.exp(-g1_logit_base))
            g1_logit_treat = -1.5 + 0.07 * age[g1_indices] + 0.8 * homeown[g1_indices]
            treat_prob[g1_indices] = 1 / (1 + np.exp(-g1_logit_treat))

        # グループ2: treatmentで成約 (y0=0, y1=1)
        g2_indices = np.where(response_group == 2)[0]
        if len(g2_indices) > 0:
            g2_logit_base = -5 + 0.04 * age[g2_indices] + 0.3 * homeown[g2_indices]
            base_prob[g2_indices] = 1 / (1 + np.exp(-g2_logit_base))
            g2_logit_treat = -1 + 0.06 * age[g2_indices] + 0.6 * homeown[g2_indices]
            treat_prob[g2_indices] = 1 / (1 + np.exp(-g2_logit_treat))

        # グループ3: treatmentで離脱 (y0=1, y1=0)
        g3_indices = np.where(response_group == 3)[0]
        if len(g3_indices) > 0:
            g3_logit_base = -1 + 0.06 * age[g3_indices] + 0.6 * homeown[g3_indices]
            base_prob[g3_indices] = 1 / (1 + np.exp(-g3_logit_base))
            g3_logit_treat = -5 + 0.03 * age[g3_indices] + 0.3 * homeown[g3_indices]
            treat_prob[g3_indices] = 1 / (1 + np.exp(-g3_logit_treat))

        # グループ4: 常に不成約 (y0=0, y1=0)
        g4_indices = np.where(response_group == 4)[0]
        if len(g4_indices) > 0:
            g4_logit_base = -6 + 0.02 * age[g4_indices] + 0.2 * homeown[g4_indices]
            base_prob[g4_indices] = 1 / (1 + np.exp(-g4_logit_base))
            g4_logit_treat = -5.5 + 0.025 * age[g4_indices] + 0.25 * homeown[g4_indices]
            treat_prob[g4_indices] = 1 / (1 + np.exp(-g4_logit_treat))

        # 6. 潜在的アウトカムを生成
        np.random.seed(self.seed + 1)
        y0 = (np.random.random(N) < base_prob).astype(int)
        np.random.seed(self.seed + 2)
        y1 = (np.random.random(N) < treat_prob).astype(int)

        # 7. 観測されるアウトカムを計算
        conversion_prob = np.where(treatment == 1, treat_prob, base_prob)
        outcome = np.where(treatment == 1, y1, y0)

        # 8. OBPのバンディットフィードバック形式に変換
        # アクションをOBP形式に変換（0/1）
        actions = treatment

        # 報酬をアウトカムとして設定
        rewards = outcome

        # 期待報酬を設定（各アクションの期待報酬）
        expected_reward = np.zeros((N, self.n_actions))
        expected_reward[:, 0] = base_prob  # アクション0（処置なし）の期待報酬
        expected_reward[:, 1] = treat_prob  # アクション1（処置あり）の期待報酬

        # 行動方策の確率を計算
        pi_b = np.zeros((N, self.n_actions))
        pi_b[:, 0] = 1 - ps  # アクション0の選択確率
        pi_b[:, 1] = ps      # アクション1の選択確率

        # propensity score（選択されたアクションの確率）
        pscore = pi_b[np.arange(N), actions]

        # 追加情報を含めたバンディットフィードバックを返す
        return dict(
            n_rounds=N,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=expected_reward,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pscore,
            # 追加情報
            age=age,
            homeown=homeown,
            response_group=response_group,
            marketing_receptivity=marketing_receptivity,
            price_sensitivity=price_sensitivity,
            base_prob=base_prob,
            treat_prob=treat_prob,
            y0=y0,
            y1=y1,
        )



# 1. 合成データの作成

n_actions = 2  # 購入するかしないかの2値
dim_context = 2  # コンテキスト（ユーザー特徴量など）の次元数

## original
#dataset = SyntheticBanditDataset(
#    n_actions=n_actions,  # クーポンを発行するかどうかの2値
#    dim_context=dim_context, # ユーザーの特徴量
#    reward_function=logistic_reward_function,
#    behavior_policy_function=linear_behavior_policy,
#    reward_type="binary",
#    random_state=12345
#)

## custom context
#dataset = CustomContextBanditDataset(
#    n_actions=n_actions,
#    dim_context=dim_context,
#    reward_function=logistic_reward_function,  # 報酬関数
#    behavior_policy_function=linear_behavior_policy,  # 行動方策
#    reward_type="binary",  # 2値の報酬（購入=1, 非購入=0）
#    random_state=12345
#)

## custom context custom policy
#dataset = CustomContextBanditDataset(
#    n_actions=n_actions,
#    dim_context=dim_context,
#    reward_function=logistic_reward_function,  # 報酬関数
#    behavior_policy_function=custom_behavior_policy,  # 行動方策
#    reward_type="binary",  # 2値の報酬（購入=1, 非購入=0）
#    random_state=12345
#)

##n_treated = 5000  # Assign treatment to top 100 users
#n_treated = None
##
#dataset = TopNTreatmentBanditDataset(
#    n_actions=n_actions,
#    dim_context=dim_context,
#    n_treated=n_treated,  # Set to None for standard probabilistic assignment
#    reward_function=logistic_reward_function,
#    behavior_policy_function=custom_behavior_policy,
#    reward_type="binary",
#    random_state=12345
#)

dataset = GroupBasedBanditDataset(
    n_actions=2,  # 処置あり/なしの2値
    dim_context=2,  # 年齢と持ち家の2次元
    seed=12345
)


# トレーニングデータとテストデータを生成
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=10000)

## 全体の反応率
bandit_feedback = bandit_feedback_test

seed = 42
total_size = bandit_feedback['n_rounds']
base_prob = bandit_feedback["expected_reward"][:,0]
treat_prob = bandit_feedback["expected_reward"][:,1]
np.random.seed(seed + 1)
y0 = (np.random.random(total_size) < base_prob).astype(int)
np.random.seed(seed + 2)
y1 = (np.random.random(total_size) < treat_prob).astype(int)

total_control_mask = bandit_feedback["action"] == 0
total_control_size = np.sum(total_control_mask)
total_control_conversion = np.sum(bandit_feedback["reward"][total_control_mask])
total_control_rate = total_control_conversion / total_control_size

total_treatment_mask = bandit_feedback["action"] == 1
total_treatment_size = np.sum(total_treatment_mask)
total_treatment_conversion = np.sum(bandit_feedback["reward"][total_treatment_mask])
total_treatment_rate = total_treatment_conversion / total_treatment_size

#ate = np.mean(bandit_feedback["y1"]) - np.mean(bandit_feedback["y0"])
ate = np.mean(y1) - np.mean(y0)
random_policy_value = np.mean(y0) + 0.5 * ate

print(f"全体(サイズ: {total_size}, ATE: {ate}), random_policy_value: {random_policy_value}:")
print(f"  総トリートメント数={total_treatment_size}")
print(f"  コントロール反応率={total_control_rate:.4f} ({total_control_conversion}/{total_control_size})")
print(f"  トリートメント反応率={total_treatment_rate:.4f} ({total_treatment_conversion}/{total_treatment_size})")
print(f"  観測されたリフト={total_treatment_rate-total_control_rate:.4f}")

original_action_dist = np.array([[[1],[0]] if a == 1 else [[0],[1]] for a in bandit_feedback["action"]])
original_ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
    expected_reward=bandit_feedback["expected_reward"],
    action_dist=original_action_dist
)
print("元の真の方策価値:", original_ground_truth_policy_value)

random_action = np.zeros(total_size)

# ランダムに2000個のインデックスを選択
#random_indices = np.random.choice(total_size, size=total_treatment_size, replace=False)
random_indices = np.random.choice(total_size, size=int(total_size / 2), replace=False)

# 選択したインデックスの位置に1を設定
random_action[random_indices] = 1

random_action_dist = np.array([[[1],[0]] if a == 1 else [[0],[1]] for a in random_action])
random_ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
    expected_reward=bandit_feedback["expected_reward"],
    action_dist=random_action_dist
)
print("ランダム方策の方策価値:", random_ground_truth_policy_value)

print("")

## バンディットフィードバックデータの生成
##bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
#bandit_feedback = bandit_feedback_test
#
## グループごとの反応率を計算
#for group_id in range(1, 5):
#    group_mask = bandit_feedback["response_group"] == group_id
#    group_size = np.sum(group_mask)
#
#    # 処置なし（コントロール）グループの反応率
#    control_mask = group_mask & (bandit_feedback["action"] == 0)
#    control_size = np.sum(control_mask)
#    control_conversion = np.sum(bandit_feedback["reward"][control_mask]) if control_size > 0 else 0
#    control_rate = control_conversion / control_size if control_size > 0 else 0
#
#    # 処置あり（トリートメント）グループの反応率
#    treatment_mask = group_mask & (bandit_feedback["action"] == 1)
#    treatment_size = np.sum(treatment_mask)
#    treatment_conversion = np.sum(bandit_feedback["reward"][treatment_mask]) if treatment_size > 0 else 0
#    treatment_rate = treatment_conversion / treatment_size if treatment_size > 0 else 0
#
#    # 潜在的アウトカム（y0, y1）の平均
#    avg_y0 = np.mean(bandit_feedback["y0"][group_mask])
#    avg_y1 = np.mean(bandit_feedback["y1"][group_mask])
#
#    sum_y0 = np.sum(bandit_feedback["y0"][group_mask])
#    sum_y1 = np.sum(bandit_feedback["y1"][group_mask])
#
#    sum_y0_digit = np.sum(bandit_feedback["expected_reward"][:,0][group_mask])
#    sum_y1_digit = np.sum(bandit_feedback["expected_reward"][:,1][group_mask])
#
#    # 結果を表示
#    print(f"グループ{group_id}（サイズ: {group_size}）:")
#    print(f"  理論値: コントロール反応率={avg_y0:.4f}, トリートメント反応率={avg_y1:.4f}, 平均リフト={avg_y1-avg_y0:.4f}, 総リフト={sum_y1-sum_y0:.4f}, 総期待報酬(実数)={sum_y1_digit-sum_y0_digit:.4f}")
#    print(f"  観測値: コントロール反応率={control_rate:.4f} ({control_conversion}/{control_size}), "
#          f"トリートメント反応率={treatment_rate:.4f} ({treatment_conversion}/{treatment_size})")
#    print()



# 2. 評価方策の定義と学習
eval_policy = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=LogisticRegression()
)
eval_policy.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
action_dist = eval_policy.predict(context=bandit_feedback_test["context"])
#action_dist = random_action_dist

## 確認
#product = (action_dist.squeeze(axis=2) * bandit_feedback["expected_reward"])
#total_sum = product.sum()
#print(f"要素ごとの掛け算の総和: {total_sum}")


# 3. 報酬モデルの学習
# ロジスティック回帰
#regression_model = RegressionModel(
#    n_actions=dataset.n_actions,
#    base_model=LogisticRegression(),
#)

#regression_model = RegressionModel(
#    n_actions=dataset.n_actions,
#    base_model=GradientBoostingClassifier(
#        n_estimators=100,
#        learning_rate=0.005,
#        max_depth=5,
#        min_samples_leaf=10,
#        random_state=12345
#    ),
#)

# ランダムフォレsと
regression_model = RegressionModel(
    n_actions=dataset.n_actions,
    base_model=RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=10,
        random_state=12345
    ),
)


estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=bandit_feedback_test["context"],
    action=bandit_feedback_test["action"],
    reward=bandit_feedback_test["reward"],
    n_folds=3,
    random_state=12345,
)

# 4. OPE手法による方策評価
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback_test,
    ope_estimators=[
        IPW(),
        DM(),
        DR(),
        SNIPW(),
        SwitchDR(),
        DoublyRobustTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="mse",
            estimator_name="dr (tuning-mse)",
        ),
        DoublyRobustTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="slope",
            estimator_name="dr (tuning-slope)",
        ),
        SelfNormalizedDoublyRobust(),
        SwitchDoublyRobustTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="mse",
            estimator_name="switch-dr (tuning-mse)",
        ),
        SwitchDoublyRobustTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="slope",
            estimator_name="switch-dr (tuning-slope)",
        ),
        DoublyRobustWithShrinkageTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="mse",
            estimator_name="dr-os (tuning-mse)",
        ),
        DoublyRobustWithShrinkageTuning(
            lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
            tuning_method="slope",
            estimator_name="dr-os (tuning-slope)",
        ),


    ]
)
estimated_policy_values = ope.estimate_policy_values(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
)
print("推定された方策価値:", estimated_policy_values)

# 5. 真の方策価値との比較
ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
    expected_reward=bandit_feedback_test["expected_reward"],
    action_dist=action_dist
)
print("真の方策価値:", ground_truth_policy_value)

# OPE手法の性能評価
ope_performance = ope.evaluate_performance_of_estimators(
    ground_truth_policy_value=ground_truth_policy_value,
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    metric="se"  # 二乗誤差を使用
)
print("OPE手法の性能評価 (SE):", ope_performance)

## 結果の可視化
#ope.visualize_off_policy_estimates(
#    action_dist=action_dist,
#    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
#)
