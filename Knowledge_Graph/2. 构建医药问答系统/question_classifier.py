import os
import ahocorasick  # 用于高效多模式字符串匹配（AC自动机）


class QuestionClassifier:
    """
    问题分类器：根据用户输入的问题，识别其中的医学实体（如疾病、症状、药品等），
    并结合关键词判断问题类型，用于后续的知识图谱查询。
    """

    def __init__(self):
        """
        初始化分类器，加载词典、构建AC自动机、定义问题关键词等。
        """

        # 获取当前文件所在目录路径（用于相对路径加载词典）
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

        # ------------------- 定义各类医学实体词典文件路径 -------------------
        self.disease_path = os.path.join(cur_dir, 'dict/disease.txt')  # 疾病词典
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')  # 科室词典
        self.check_path = os.path.join(cur_dir, 'dict/check.txt')  # 检查项目词典
        self.drug_path = os.path.join(cur_dir, 'dict/drug.txt')  # 药品词典
        self.food_path = os.path.join(cur_dir, 'dict/food.txt')  # 食物词典
        self.producer_path = os.path.join(cur_dir, 'dict/producer.txt')  # 药品厂商词典
        self.symptom_path = os.path.join(cur_dir, 'dict/symptom.txt')  # 症状词典
        self.deny_path = os.path.join(cur_dir, 'dict/deny.txt')  # 否定词词典（如“不”、“忌”）

        # ------------------- 加载各类特征词 -------------------
        # 每个词表读取为列表，去除空行和首尾空格
        self.disease_wds = [i.strip() for i in open(self.disease_path, encoding='utf-8') if i.strip()]
        self.department_wds = [i.strip() for i in open(self.department_path, encoding='utf-8') if i.strip()]
        self.check_wds = [i.strip() for i in open(self.check_path, encoding='utf-8') if i.strip()]
        self.drug_wds = [i.strip() for i in open(self.drug_path, encoding='utf-8') if i.strip()]
        self.food_wds = [i.strip() for i in open(self.food_path, encoding='utf-8') if i.strip()]
        self.producer_wds = [i.strip() for i in open(self.producer_path, encoding='utf-8') if i.strip()]
        self.symptom_wds = [i.strip() for i in open(self.symptom_path, encoding='utf-8') if i.strip()]
        self.deny_words = [i.strip() for i in open(self.deny_path, encoding='utf-8') if i.strip()]

        # 所有医学相关词汇的并集（用于实体识别）
        self.region_words = set(
            self.disease_wds + self.department_wds + self.check_wds +
            self.drug_wds + self.food_wds + self.producer_wds + self.symptom_wds
        )

        # ------------------- 构建AC自动机（用于快速实体识别） -------------------
        # AC自动机可高效匹配文本中所有出现的医学词汇
        self.region_tree = self.build_actree(list(self.region_words))

        # ------------------- 构建词 -> 类型映射字典 -------------------
        # 例如：{'感冒': ['disease'], '头痛': ['symptom'], '阿司匹林': ['drug']}
        self.wdtype_dict = self.build_wdtype_dict()

        # ------------------- 定义各类问题的疑问词（关键词） -------------------
        # 根据这些关键词判断用户想问什么类型的问题

        # 症状类问题关键词
        self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']

        # 病因类问题关键词
        self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何',
                           '如何才会', '怎么才会', '会导致', '会造成']

        # 并发症类问题关键词
        self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现',
                              '伴随发生', '伴随', '共现']

        # 饮食类问题关键词
        self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱',
                          '菜谱', '食用', '食物', '补品']

        # 药品类问题关键词
        self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']

        # 预防类问题关键词
        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开',
                             '避掉',
                             '躲开', '躲掉', '绕开',
                             '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不',
                             '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不',
                             '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                             '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']

        # 治疗周期类问题关键词
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时',
                              '几个小时', '多少年']

        # 治疗方式类问题关键词
        self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法',
                             '咋治', '怎么办', '咋办', '咋治']

        # 治愈概率类问题关键词
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性',
                              '能治', '可治', '可以治', '可以医']

        # 易感人群类问题关键词
        self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']

        # 检查类问题关键词
        self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']

        # 科室归属类问题关键词
        self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']

        # “能治什么病”类问题关键词（用于药品/检查反向查询）
        self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用',
                          '用处', '用途', '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚',
                          '需要', '要']

        print('model init finished ......')

    def classify(self, question):
        """
        主分类函数：接收用户问题，识别实体并分类问题类型。
        Args:
            question (str): 用户输入的问题文本
        Returns:
            dict: 包含识别出的实体和问题类型的字典，格式如下：
                  {
                    'args': {'感冒': ['disease'], '头痛': ['symptom']},
                    'question_types': ['disease_symptom']
                  }
        """
        data = {}
        # 第一步：识别问题中出现的医学实体
        medical_dict = self.check_medical(question)

        # 如果没有识别出任何医学实体，返回空结果
        if not medical_dict:
            return {}

        data['args'] = medical_dict  # 存储识别出的实体及其类型

        # 收集所有识别出的实体类型（如 disease, symptom 等）
        types = []
        for type_list in medical_dict.values():
            types.extend(type_list)
        types = list(set(types))  # 去重

        question_types = []  # 存储所有可能的问题类型

        # -------------------- 分类逻辑：根据关键词+实体类型判断问题意图 --------------------

        # 1. 疾病相关症状
        if self.check_words(self.symptom_qwds, question) and ('disease' in types):
            question_types.append('disease_symptom')

        # 2. 症状对应哪些疾病
        if self.check_words(self.symptom_qwds, question) and ('symptom' in types):
            question_types.append('symptom_disease')

        # 3. 疾病的病因
        if self.check_words(self.cause_qwds, question) and ('disease' in types):
            question_types.append('disease_cause')

        # 4. 疾病的并发症
        if self.check_words(self.acompany_qwds, question) and ('disease' in types):
            question_types.append('disease_acompany')

        # 5. 疾病宜吃/忌吃食物（需结合否定词判断）
        if self.check_words(self.food_qwds, question) and ('disease' in types):
            deny_status = self.check_words(self.deny_words, question)
            question_type = 'disease_not_food' if deny_status else 'disease_do_food'
            question_types.append(question_type)

        # 6. 食物对应哪些疾病（推荐/禁忌）
        if self.check_words(self.food_qwds + self.cure_qwds, question) and ('food' in types):
            deny_status = self.check_words(self.deny_words, question)
            question_type = 'food_not_disease' if deny_status else 'food_do_disease'
            question_types.append(question_type)

        # 7. 疾病推荐药品
        if self.check_words(self.drug_qwds, question) and ('disease' in types):
            question_types.append('disease_drug')

        # 8. 药品能治什么病
        if self.check_words(self.cure_qwds, question) and ('drug' in types):
            question_types.append('drug_disease')

        # 9. 疾病需要做哪些检查
        if self.check_words(self.check_qwds, question) and ('disease' in types):
            question_types.append('disease_check')

        # 10. 检查项目用于诊断哪些疾病
        if self.check_words(self.check_qwds + self.cure_qwds, question) and ('check' in types):
            question_types.append('check_disease')

        # 11. 疾病如何预防
        if self.check_words(self.prevent_qwds, question) and ('disease' in types):
            question_types.append('disease_prevent')

        # 12. 疾病治疗周期
        if self.check_words(self.lasttime_qwds, question) and ('disease' in types):
            question_types.append('disease_lasttime')

        # 13. 疾病治疗方式
        if self.check_words(self.cureway_qwds, question) and ('disease' in types):
            question_types.append('disease_cureway')

        # 14. 疾病治愈概率
        if self.check_words(self.cureprob_qwds, question) and ('disease' in types):
            question_types.append('disease_cureprob')

        # 15. 疾病易感人群
        if self.check_words(self.easyget_qwds, question) and ('disease' in types):
            question_types.append('disease_easyget')

        # 16. 默认返回疾病描述（未匹配具体问题类型但包含疾病）
        if not question_types and 'disease' in types:
            question_types = ['disease_desc']

        # 17. 默认返回症状相关疾病（仅提到症状）
        if not question_types and 'symptom' in types:
            question_types = ['symptom_disease']

        # ------------------- 返回最终分类结果 -------------------
        data['question_types'] = question_types
        print("分类结果:", data)
        return data

    def build_wdtype_dict(self):
        """
        构建词汇到类型的映射字典。
        一个词可能属于多个类型（如“感冒药”既是药也是症状相关词，但此处按精确匹配）。
        Returns:
            dict: 如 {'感冒': ['disease'], '头痛': ['symptom'], '阿司匹林': ['drug']}
        """
        wd_dict = {}
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.disease_wds:
                wd_dict[wd].append('disease')
            if wd in self.department_wds:
                wd_dict[wd].append('department')
            if wd in self.check_wds:
                wd_dict[wd].append('check')
            if wd in self.drug_wds:
                wd_dict[wd].append('drug')
            if wd in self.food_wds:
                wd_dict[wd].append('food')
            if wd in self.symptom_wds:
                wd_dict[wd].append('symptom')
            if wd in self.producer_wds:
                wd_dict[wd].append('producer')
        return wd_dict

    def build_actree(self, wordlist):
        """
        使用 ahocorasick 构建AC自动机，实现高效多关键词匹配。
        Args:
            wordlist (list): 词汇列表
        Returns:
            ahocorasick.Automaton: 构建好的AC自动机对象
        """
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))  # (word, value) 存储索引和词本身
        actree.make_automaton()  # 构建失败指针，完成自动机构建
        return actree

    def check_medical(self, question):
        """
        使用AC自动机从问题中提取所有医学实体，并去除包含关系中的子串（如“肺炎”和“病毒性肺炎”共现时保留长的）。
        Args:
            question (str): 用户问题
        Returns:
            dict: 识别出的实体及其类型，如 {'肺炎': ['disease'], '头痛': ['symptom']}
        """
        region_wds = []  # 存储所有匹配到的词
        for match in self.region_tree.iter(question):  # AC自动机匹配
            word = match[1][1]  # 提取匹配到的词
            region_wds.append(word)

        # 去除包含关系中的短词（避免“肺炎”和“病毒性肺炎”同时出现）
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:  # wd1 被 wd2 包含
                    stop_wds.append(wd1)
        final_wds = [wd for wd in region_wds if wd not in stop_wds]  # 过滤后保留长词

        # 构建最终实体 -> 类型字典
        final_dict = {wd: self.wdtype_dict.get(wd, []) for wd in final_wds}
        return final_dict

    def check_words(self, wds, sent):
        """
        判断句子中是否包含任一关键词。
        Args:
            wds (list): 关键词列表
            sent (str): 句子
        Returns:
            bool: 是否包含
        """
        for wd in wds:
            if wd in sent:
                return True
        return False


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    """
    测试用例：启动一个交互式命令行，输入问题并输出分类结果。
    """
    handler = QuestionClassifier()
    while True:
        question = input('请输入问题')
        data = handler.classify(question)
        print("分类结果:", data)