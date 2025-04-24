# interest_evolution.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from helper import choice_model
from helper import document
from helper import user
from helper import utils
from helper import environment
from helper import rec_gym

# 定义了用户对视频的交互反馈：是否点击/观看时间/点赞/内容质量/视频类别
class IEvResponse(user.AbstractResponse):
    MIN_QUALITY_SCORE = -100
    MAX_QUALITY_SCORE = 100

    def __init__(self, clicked=False, watch_time=0.0, quality=0.0, cluster_id=0.0):
        self.clicked = clicked
        self.watch_time = watch_time
        self.quality = quality
        self.cluster_id = cluster_id

    def create_observation(self):
        return {
            'click': int(self.clicked),
            'watch_time': np.array(self.watch_time),
            'quality': np.array(self.quality),
            'cluster_id': int(self.cluster_id)
        }
    @classmethod
    def response_space(cls):
        return spaces.Dict({
            'click': spaces.Discrete(2), 
            'watch_time': spaces.Box(0.0, IEvVideo.MAX_VIDEO_LENGTH, shape=(), dtype=np.float32),
            'quality': spaces.Box(cls.MIN_QUALITY_SCORE, cls.MAX_QUALITY_SCORE, shape=(), dtype=np.float32),
            'cluster_id': spaces.Discrete(IEvVideo.NUM_FEATURES)
        })

# 视频对象: 
class IEvVideo(document.AbstractDocument):
    MAX_VIDEO_LENGTH = 100.0
    # 视频的特征维度
    NUM_FEATURES = 7

    def __init__(self, doc_id, features, cluster_id=None, video_length=None, quality=None):
        # one-hot encoding = [0,0,1,0,0,0,0]
        self.features = features
        # cluster_id = 2
        self.cluster_id = cluster_id
        # 外部构造时显示传入
        self.video_length = video_length
        self.quality = quality
        super().__init__(doc_id)

    def create_observation(self):
        return self.features

    @classmethod
    def observation_space(cls):
        return spaces.Box(shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)


class UtilityModelVideoSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_ctor=IEvVideo, min_utility=-3.0, max_utility=3.0, video_length=4.0, **kwargs):
        super().__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self._num_clusters = self.get_doc_ctor().NUM_FEATURES
        self._min_utility = min_utility
        self._max_utility = max_utility
        self._video_length = video_length
        num_trashy = int(round(self._num_clusters * 0.7))
        num_nutritious = self._num_clusters - num_trashy

        trashy = np.linspace(self._min_utility, 0, num_trashy, endpoint=False)
        nutritious = np.linspace(0, self._max_utility, num_nutritious)

        self.cluster_means = np.concatenate((trashy, nutritious))

    def sample_document(self):
        cluster_id = self._rng.integers(0, self._num_clusters)
        features = np.zeros(self._num_clusters)
        features[cluster_id] = 1.0
        quality_mean = self.cluster_means[cluster_id]
        quality = self._rng.normal(quality_mean, 0.1)

        doc_features = {
            'doc_id': self._doc_count,
            'features': features,
            'cluster_id': cluster_id,
            'video_length': self._video_length,
            'quality': quality
        }

        self._doc_count += 1
        return self._doc_ctor(**doc_features)   


class IEvUserState(user.AbstractUserState):
    NUM_FEATURES = 7

    def __init__(self, user_interests, time_budget=None, score_scaling=None, attention_prob=None, no_click_mass=None, user_update_alpha=None, step_penalty=None, min_normalizer=None,
                 user_quality_factor=None, document_quality_factor=None):
        self.user_interests = user_interests
        self.time_budget = time_budget
        self.choice_features = {
            'score_scaling': score_scaling,
            'attention_prob': attention_prob,
            'no_click_mass': no_click_mass,
            'min_normalizer': min_normalizer
        }
        self.user_update_alpha = user_update_alpha
        self.step_penalty = step_penalty
        self.user_quality_factor = user_quality_factor
        self.document_quality_factor = document_quality_factor

    def score_document(self, doc_obs):
        if self.user_interests.shape != doc_obs.shape:
            print("User dimension: ",self.user_interests.shape, "Doc_obs dimension: ", doc_obs.shape)
            raise ValueError('User and document feature dimension mismatch!')
        return np.dot(self.user_interests, doc_obs)

    def create_observation(self):
        return self.user_interests

    @classmethod
    def observation_space(cls):
        return spaces.Box(shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)

class UtilityModelUserSampler(user.AbstractUserSampler):
    """Class that samples users for utility model experiment."""

    def __init__(self,
                 user_ctor=IEvUserState,
                 document_quality_factor=1.0,
                 no_click_mass=1,
                 min_normalizer=-1.0,
                 **kwargs):
        self._no_click_mass = no_click_mass
        self._min_normalizer = min_normalizer
        self._document_quality_factor = document_quality_factor
        super().__init__(user_ctor=user_ctor, **kwargs)

    def sample_user(self):
        features = {
            'user_interests': self._rng.uniform(-1.0, 1.0, self.get_user_ctor().NUM_FEATURES),
            'time_budget': 200.0,
            'no_click_mass': self._no_click_mass,
            'step_penalty': 0.5,
            'score_scaling': 0.05,
            'attention_prob': 0.65,
            'min_normalizer': self._min_normalizer,
            'user_quality_factor': 0.0,
            'document_quality_factor': self._document_quality_factor,
            'user_update_alpha': 0.9 * (1.0 / 3.4)
        }
        return self._user_ctor(**features)

class IEvUserModel(user.AbstractUserModel):
    def __init__(self, slate_size, choice_model_ctor, response_model_ctor=IEvResponse,
                 user_state_ctor=IEvUserState, no_click_mass=0.5, seed=0,
                 alpha_x_intercept=1.0, alpha_y_intercept=0.3):
        super().__init__(
            response_model_ctor,
            UtilityModelUserSampler(
            user_ctor=user_state_ctor, no_click_mass=no_click_mass, seed=seed),
        slate_size)
        if choice_model_ctor is None:
            raise Exception('A choice model needs to be specified!')
        self.choice_model = choice_model_ctor(self._user_state.choice_features)
        self._alpha_x_intercept = alpha_x_intercept
        self._alpha_y_intercept = alpha_y_intercept

    def is_terminal(self):
        return self._user_state.time_budget <= 0

    def update_state(self, slate_documents, responses):
        def compute_alpha(x, x_int, y_int):
            return (-y_int / x_int) * np.abs(x) + y_int

        user_state = self._user_state
        for doc, response in zip(slate_documents, responses):
            if response.clicked:
                self.choice_model.score_documents(user_state, [doc.create_observation()])
                expected_utility = self.choice_model.scores[0]
                target = doc.features - user_state.user_interests
                mask = doc.features
                alpha = compute_alpha(user_state.user_interests, self._alpha_x_intercept, self._alpha_y_intercept)
                update = alpha * mask * target
                prob = np.dot((user_state.user_interests + 1.0) / 2, mask)
                if np.random.rand(1) < prob:
                    user_state.user_interests += update
                else:
                    user_state.user_interests -= update
                user_state.user_interests = np.clip(user_state.user_interests, -1.0, 1.0)

                received_utility = (
                    user_state.user_quality_factor * expected_utility +
                    user_state.document_quality_factor * doc.quality
                )
                user_state.time_budget -= response.watch_time
                user_state.time_budget += (
                    user_state.user_update_alpha * response.watch_time * received_utility
                )
                return

        user_state.time_budget -= user_state.step_penalty

    def simulate_response(self, documents):
        responses = [self._response_model_ctor() for _ in documents]
        doc_obs = [doc.create_observation() for doc in documents]
        self.choice_model.score_documents(self._user_state, doc_obs)
        selected_index = self.choice_model.choose_item()

        for i, response in enumerate(responses):
            response.quality = documents[i].quality
            response.cluster_id = documents[i].cluster_id

        if selected_index is not None:
            self._generate_click_response(documents[selected_index], responses[selected_index])

        return responses

    def _generate_click_response(self, doc, response):
        user_state = self._user_state
        response.clicked = True
        response.watch_time = min(user_state.time_budget, doc.video_length)

# Reward function
def clicked_watchtime_reward(responses):
    return sum(r.watch_time for r in responses if r.clicked)


# def total_clicks_reward(responses):
#     return sum(int(r.clicked) for r in responses)


def create_environment(env_config):
    user_model = IEvUserModel(
        env_config['slate_size'],
        choice_model_ctor=choice_model.MultinomialProportionalChoiceModel,
        response_model_ctor=IEvResponse,
        user_state_ctor=IEvUserState,
        seed=env_config['seed']
    )
    document_sampler = UtilityModelVideoSampler(
        doc_ctor=IEvVideo,
        seed=env_config['seed']
    )

    env = environment.Environment(
        user_model,
        document_sampler,
        # 10 个 doc
        env_config['num_candidates'],
        # 调2个推荐
        env_config['slate_size'],
        resample_documents=env_config['resample_documents']
    )

    return rec_gym.RecSimGymEnv(
        env,
        # watch time as a reward
        clicked_watchtime_reward,
        # metrics: impression, click, quality, cluster_watch_count_no_click
        utils.aggregate_video_cluster_metrics,
        # metrics: 'CTR'= clicks / impressions; 'AverageQuality'= quality / clicks
        utils.write_video_cluster_metrics
    )