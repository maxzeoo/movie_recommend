import math
import random
from collections import defaultdict, Counter
import time

# 计时装饰器
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 运行时间: {end_time - start_time:.4f}s")
        return res
    return wrapper

# 数据集类：加载评分+用户属性+电影类型数据（修复编码问题）
class Dataset:
    def __init__(self, ratings_path, users_path, movies_path):
        self.ratings_path = ratings_path  # 评分数据路径
        self.users_path = users_path      # 用户属性路径
        self.movies_path = movies_path    # 电影类型路径
        self.data = self.load_ratings()   # 评分数据：{user: {item: rating}}
        self.user_profile = self.load_user_profile()  # 用户属性：{user: {gender, age, occupation}}
        self.movie_genres = self.load_movie_genres()  # 电影类型：{movie_id: [genres]}

    def load_ratings(self):
        """加载ratings.dat评分数据"""
        data = defaultdict(dict)
        with open(self.ratings_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user_id, movie_id, rating, _ = line.split('::')
                user_id = int(user_id)
                movie_id = int(movie_id)
                rating = float(rating)
                data[user_id][movie_id] = rating
        return data

    def load_user_profile(self):
        """加载users.dat用户属性数据（性别+年龄+职业）"""
        user_profile = defaultdict(dict)
        age_map = {1: 9, 18: 21, 25: 29.5, 35: 39.5, 45: 47, 50: 52.5, 56: 60}  # 年龄区间中点
        with open(self.users_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user_id, gender, age, occupation, _ = line.split('::')
                user_id = int(user_id)
                user_profile[user_id] = {
                    'gender': gender,
                    'age': age_map[int(age)],
                    'occupation': int(occupation)
                }
        return user_profile

    def load_movie_genres(self):
        """加载movies.dat电影类型数据（修复编码问题：UTF-8→GBK）"""
        movie_genres = defaultdict(list)
        # 关键修改：将encoding='utf-8'改为encoding='gbk'，适配Windows系统编码
        with open(self.movies_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式：MovieID::Title::Genres（Genres用|分隔）
                parts = line.split('::')
                if len(parts) < 3:
                    continue  # 跳过异常行
                movie_id = int(parts[0])
                genres = parts[2].split('|')
                movie_genres[movie_id] = genres
        return movie_genres

    def splitData(self, M=8, k=0):
        """M折交叉验证切分训练集/测试集"""
        train = defaultdict(dict)
        test = defaultdict(dict)
        for user, items in self.data.items():
            for item, rating in items.items():
                if random.randint(0, M - 1) == k:
                    test[user][item] = rating
                else:
                    train[user][item] = rating
        return train, test

# 评估指标类（无修改）
class Metric:
    def __init__(self, train, test, recs):
        self.train = train
        self.test = test
        self.recs = recs
        self.all_items = self.get_all_items()

    def get_all_items(self):
        all_items = set()
        for user_items in self.train.values():
            all_items.update(user_items.keys())
        for user_items in self.test.values():
            all_items.update(user_items.keys())
        return all_items

    def precision(self):
        hit = 0
        all_rec = 0
        for user in self.test:
            test_items = set(self.test[user].keys())
            rec_items = [item for item, _ in self.recs.get(user, [])]
            hit += len(test_items & set(rec_items))
            all_rec += len(rec_items)
        return round(hit / all_rec * 100, 2) if all_rec > 0 else 0.0

    def recall(self):
        hit = 0
        all_test = 0
        for user in self.test:
            test_items = set(self.test[user].keys())
            rec_items = [item for item, _ in self.recs.get(user, [])]
            hit += len(test_items & set(rec_items))
            all_test += len(test_items)
        return round(hit / all_test * 100, 2) if all_test > 0 else 0.0

    def coverage(self):
        rec_items = set()
        for user_recs in self.recs.values():
            rec_items.update([item for item, _ in user_recs])
        return round(len(rec_items) / len(self.all_items) * 100, 2) if self.all_items else 0.0

    def popularity(self):
        item_pop = Counter()
        for user_items in self.train.values():
            item_pop.update(user_items.keys())
        pop_sum = 0
        rec_count = 0
        for user_recs in self.recs.values():
            for item, _ in user_recs:
                pop_sum += math.log(1 + item_pop.get(item, 0))
                rec_count += 1
        return round(pop_sum / rec_count, 4) if rec_count > 0 else 0.0

    def eval(self):
        return {
            '精确率': self.precision(),
            '召回率': self.recall(),
            '覆盖率': self.coverage(),
            '新颖度': self.popularity()
        }

# 基于用户的协同过滤算法（融合行为+属性+电影类型）
class UserCF:
    def __init__(self, train, user_profile, movie_genres,
                 K=20, N=10, alpha=0.7, beta=0.2, gamma=0.1,
                 attr_weights={'gender': 0.4, 'age': 0.3, 'occupation': 0.3}):
        self.train = train                # 训练集评分数据
        self.user_profile = user_profile  # 用户属性数据
        self.movie_genres = movie_genres  # 电影类型数据
        self.K = K                        # TopK相似用户
        self.N = N                        # TopN推荐物品
        # 可配置权重参数
        self.alpha = alpha                # 行为相似度权重
        self.beta = beta                  # 用户属性相似度权重
        self.gamma = gamma                # 电影类型匹配度权重
        self.attr_weights = attr_weights  # 用户属性内部权重（性别:年龄:职业）
        # 预计算用户类型偏好（训练时一次性计算）
        self.user_genre_preference = self.calculate_user_genre_preference()
        # 最终用户相似度矩阵
        self.user_sim = self.calculate_user_similarity()

    def calculate_user_genre_preference(self):
        """计算用户的电影类型偏好系数（基于训练集评分）"""
        user_genre_preference = defaultdict(dict)  # {user: {genre: 偏好系数}}
        genre_count = defaultdict(int)             # 每个用户的类型总评分次数
        genre_total_rating = defaultdict(float)    # 每个用户的类型总评分

        for user, items in self.train.items():
            for movie_id, rating in items.items():
                genres = self.movie_genres.get(movie_id, [])
                for genre in genres:
                    genre_count[(user, genre)] += 1
                    genre_total_rating[(user, genre)] += rating

        # 计算偏好系数：（类型平均评分）×（类型评分占比）
        for user, items in self.train.items():
            total_rating_count = len(items)  # 用户总评分次数
            if total_rating_count == 0:
                continue
            # 统计该用户的所有类型
            user_genres = set()
            for movie_id in items:
                user_genres.update(self.movie_genres.get(movie_id, []))
            # 计算每个类型的偏好系数
            for genre in user_genres:
                count = genre_count.get((user, genre), 0)
                total_rating = genre_total_rating.get((user, genre), 0)
                avg_rating = total_rating / count if count > 0 else 0.0  # 类型平均评分
                ratio = count / total_rating_count                        # 类型评分占比
                preference = avg_rating * ratio                           # 最终偏好系数
                user_genre_preference[user][genre] = preference

        return user_genre_preference

    def calculate_genre_match_score(self, user, movie_id):
        """计算用户与电影的类型匹配度（0~1区间）"""
        user_preferences = self.user_genre_preference.get(user, {})
        movie_genres = self.movie_genres.get(movie_id, [])
        if not user_preferences or not movie_genres:
            return 0.0  # 无偏好或无类型数据，匹配度为0
        # 计算匹配度：用户偏好类型与电影类型的交集偏好系数之和
        match_score = 0.0
        for genre in movie_genres:
            match_score += user_preferences.get(genre, 0.0)
        # 归一化到0~1区间（除以用户最大偏好系数和，避免溢出）
        max_preference_sum = sum(user_preferences.values()) if user_preferences else 1.0
        return match_score / max_preference_sum

    def calculate_behavior_similarity(self):
        """计算用户行为相似度（基于评分共现的余弦相似度）"""
        item_users = defaultdict(set)
        for user, items in self.train.items():
            for item in items:
                item_users[item].add(user)
        cooccur = defaultdict(int)
        user_item_count = defaultdict(int)
        for item, users in item_users.items():
            for u in users:
                user_item_count[u] += 1
                for v in users:
                    if u < v:
                        cooccur[(u, v)] += 1
        behavior_sim = defaultdict(dict)
        for (u, v), count in cooccur.items():
            sim = count / math.sqrt(user_item_count[u] * user_item_count[v])
            behavior_sim[u][v] = sim
            behavior_sim[v][u] = sim
        return behavior_sim

    def calculate_attribute_similarity(self):
        """计算用户属性相似度（融合性别+年龄+职业，支持自定义内部权重）"""
        attribute_sim = defaultdict(dict)
        all_users = set(self.train.keys()) & set(self.user_profile.keys())
        max_age_diff = 60 - 9  # 年龄区间中点最大差值（60-9=51）
        g_weight = self.attr_weights['gender']
        a_weight = self.attr_weights['age']
        o_weight = self.attr_weights['occupation']

        for u in all_users:
            for v in all_users:
                if u >= v:
                    continue
                # 性别相似度（0/1）
                gender_sim = 1.0 if self.user_profile[u]['gender'] == self.user_profile[v]['gender'] else 0.0
                # 职业相似度（0/1）
                occ_sim = 1.0 if self.user_profile[u]['occupation'] == self.user_profile[v]['occupation'] else 0.0
                # 年龄相似度（0~1）
                age_diff = abs(self.user_profile[u]['age'] - self.user_profile[v]['age'])
                age_sim = 1.0 - (age_diff / max_age_diff) if max_age_diff != 0 else 0.0
                # 加权计算属性总相似度（内部权重可配置）
                total_attr_sim = (g_weight * gender_sim) + (a_weight * age_sim) + (o_weight * occ_sim)
                attribute_sim[u][v] = total_attr_sim
                attribute_sim[v][u] = total_attr_sim
        return attribute_sim

    def calculate_user_similarity(self):
        """融合行为相似度与属性相似度，得到最终用户相似度"""
        behavior_sim = self.calculate_behavior_similarity()
        attribute_sim = self.calculate_attribute_similarity()
        final_sim = defaultdict(dict)
        all_users = set(self.train.keys())

        for u in all_users:
            for v in all_users:
                if u == v:
                    continue
                b_sim = behavior_sim.get(u, {}).get(v, 0.0)
                a_sim = attribute_sim.get(u, {}).get(v, 0.0)
                # 加权融合
                final_sim[u][v] = self.alpha * b_sim + self.beta * a_sim
        return final_sim

    def get_recommendation(self):
        """生成推荐列表（融合用户相似度+电影类型匹配度）"""
        recs = defaultdict(list)
        # 预计算所有物品的平均评分（用于基础得分）
        item_avg_rating = defaultdict(float)
        item_rating_count = defaultdict(int)
        for user, items in self.train.items():
            for item, rating in items.items():
                item_avg_rating[item] += rating
                item_rating_count[item] += 1
        for item in item_avg_rating:
            item_avg_rating[item] /= item_rating_count[item]

        for user in self.train:
            # 筛选TopK相似用户
            similar_users = sorted(
                [(v, sim) for v, sim in self.user_sim.get(user, {}).items() if sim > 0],
                key=lambda x: x[1],
                reverse=True
            )[:self.K]
            # 计算物品推荐得分
            item_score = defaultdict(float)
            for sim_user, sim_score in similar_users:
                for item, rating in self.train[sim_user].items():
                    if item in self.train[user]:
                       continue  # 不推荐用户已评分物品
                    # 1. 基础得分：相似用户评分 × 用户相似度
                    base_score = sim_score * rating
                    # 2. 类型匹配得分：类型匹配度 × 物品平均评分
                    genre_score = self.calculate_genre_match_score(user, item) * item_avg_rating.get(item, 3.0)
                    # 3. 最终得分：基础得分（alpha+beta权重） + 类型匹配得分（gamma权重）
                    item_score[item] += base_score + (self.gamma * genre_score)
            # 生成TopN推荐
            rec_items = sorted(item_score.items(), key=lambda x: x[1], reverse=True)[:self.N]
            recs[user] = rec_items
        return recs

# 实验类（支持所有权重参数配置）
class Experiment:
    def __init__(self, M=8, K=20, N=10,
                 ratings_fp='C:/Users/lx/Desktop/算法/ml-1m/ml-1m/ratings.dat',
                 users_fp='C:/Users/lx/Desktop/算法/ml-1m/ml-1m/users.dat',
                 movies_fp='C:/Users/lx/Desktop/算法/ml-1m/ml-1m/movies.dat',
                 rt='UserCF', alpha=0.6, beta=0.3, gamma=0.1,
                 attr_weights={'gender': 0.4, 'age': 0.3, 'occupation': 0.3}):
        self.M = M                  # 实验折数
        self.K = K                  # TopK相似用户
        self.N = N                  # TopN推荐物品
        # 文件路径
        self.ratings_fp = ratings_fp
        self.users_fp = users_fp
        self.movies_fp = movies_fp
        # 算法类型
        self.rt = rt
        # 可配置权重参数
        self.alpha = alpha          # 行为相似度权重
        self.beta = beta            # 属性相似度权重
        self.gamma = gamma          # 类型匹配度权重
        self.attr_weights = attr_weights  # 用户属性内部权重
        # 算法映射
        self.alg = {'UserCF': self.user_cf_recommend}

    def user_cf_recommend(self, train, user_profile, movie_genres):
        """UserCF推荐接口（接收所有必要参数）"""
        user_cf = UserCF(
            train=train,
            user_profile=user_profile,
            movie_genres=movie_genres,
            K=self.K,
            N=self.N,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            attr_weights=self.attr_weights
        )
        return user_cf.get_recommendation()

    @timmer
    def worker(self, train, test, user_profile, movie_genres):
        """单次实验（传递所有数据）"""
        recs = self.alg[self.rt](train, user_profile, movie_genres)
        metric = Metric(train, test, recs)
        return metric.eval()

    @timmer
    def run(self):
        """运行M次实验并计算平均值"""
        metrics = {f'精确率': 0, '召回率': 0, '覆盖率': 0, '新颖度': 0}
        # 加载完整数据集（评分+用户属性+电影类型）
        dataset = Dataset(
            ratings_path=self.ratings_fp,
            users_path=self.users_fp,
            movies_path=self.movies_fp
        )

        for ii in range(self.M):
            print(f'\n实验 {ii + 1}/{self.M}:')
            # 切分训练集和测试集
            train, test = dataset.splitData(M=self.M, k=ii)
            data=dataset.data
            user_profile = dataset.user_profile
            movie_genres = dataset.movie_genres
            print(f'训练集用户数: {len(train)}, 测试集用户数: {len(test)}')
            # 执行单次实验
            metric = self.worker(train, test, user_profile, movie_genres)
            print(f'单次实验结果: {metric}')
            # 累加指标
            for k in metrics:
                metrics[k] += metric[k]

        # 计算平均指标
        for k in metrics:
            if k == 'Popularity':
                metrics[k] = round(metrics[k] / self.M, 4)
            else:
                metrics[k] = round(metrics[k] / self.M, 2)

        # 输出最终结果（显示所有权重配置）
        print(f'\n{"=" * 80}')
        print(f'平均结果 (M={self.M}, K={self.K}, N={self.N})')
        print(f'权重配置：alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}')
        print(f'用户属性内部权重：性别={self.attr_weights["gender"]}, 年龄={self.attr_weights["age"]}, 职业={self.attr_weights["occupation"]}')
        for k, v in metrics.items():
            print(f'{k}: {v}')
        print(f'{"=" * 80}')

# 主函数：可直接运行实验，支持调整所有权重参数
if __name__ == '__main__':
    # 基础实验参数配置
    M = 8  # 8折交叉验证
    N = 10  # 推荐10个物品
    # 可配置权重参数（重点！可修改以下参数进行实验验证）
    alpha = 0.6    # 行为相似度权重（0.5~0.8可调）
    beta = 0.3     # 用户属性相似度权重（0.1~0.4可调）
    gamma = 0.1    # 电影类型匹配度权重（0.0~0.2可调）
    # 用户属性内部权重（可调整）
    attr_weights = {
        'gender': 0.4,
        'age': 0.3,
        'occupation': 0.3
    }
    # 测试不同的相似用户数K（任务书要求的K值范围）
    for K in [5, 10, 20, 40, 80, 160]:
        print(f'\n{"=" * 90}')
        print(f'开始实验：K={K}, N={N}, M={M} | 权重(alpha={alpha}, beta={beta}, gamma={gamma})')
        print(f'{"=" * 90}')
        exp = Experiment(
            M=M, K=K, N=N,
            alpha=alpha, beta=beta, gamma=gamma,
            attr_weights=attr_weights
        )
        exp.run()