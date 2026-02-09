import pymongo
from lxml import etree
import os
import json  # 新增：用于读取 JSON 文件
from max_cut import *


class MedicalGraph:
    def __init__(self):
        # self.conn = pymongo.MongoClient()  # 注释掉 MongoDB 连接
        # self.db = self.conn['medical']
        # self.col = self.db['data']

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        print(cur_dir)
        # 从 JSON 文件加载数据
        json_path = os.path.join(cur_dir, 'diseases.json')  # 可自定义文件名
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据文件: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)  # 加载整个数据列表

        # 其他初始化不变
        first_words = [
            '',  # 空字符串，防止索引越界
            '我',
            '你',
            '他',
            '她',
            '它',
            '我们',
            '你们',
            '他们',
            '这',
            '那',
            '哪',
            '谁',
            '怎',
            '如何',
            '为什么',
            '是否',
            '请',
            '建议',
            '可能',
            '常见',
            '出现',
            '患有',
            '导致',
            '引起',
            '属于',
            '一种',
            '症状',
            '疾病',
            '病人',
            '患者',
            '临床',
            '表现',
            '包括',
            '伴有',
            '并发',
            '和',
            '或',
            '与',
            '及',
            '等'
        ]
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z']
        nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        self.stop_words = first_words + alphabets + nums

        self.key_dict = {
            '医保疾病': 'yibao_status',
            "患病比例": "get_prob",
            "易感人群": "easy_get",
            "传染方式": "get_way",
            "就诊科室": "cure_department",
            "治疗方式": "cure_way",
            "治疗周期": "cure_lasttime",
            "治愈率": "cured_prob",
            '药品明细': 'drug_detail',
            '药品推荐': 'recommand_drug',
            '推荐': 'recommand_eat',
            '忌食': 'not_eat',
            '宜食': 'do_eat',
            '症状': 'symptom',
            '检查': 'check',
            '成因': 'cause',
            '预防措施': 'prevent',
            '所属类别': 'category',
            '简介': 'desc',
            '名称': 'name',
            '常用药品': 'common_drug',
            '治疗费用': 'cost_money',
            '并发症': 'acompany'
        }
        self.cuter = CutWords()

    def collect_medical(self):
        cates = []
        inspects = []
        count = 0

        # === 修改点：遍历 self.data_list 而不是 MongoDB 的 cursor ===
        for item in self.data_list:  # 直接遍历 JSON 数据
            data = {}
            basic_info = item['basic_info']
            name = basic_info.get('name')
            if not name:
                continue

            # 基本信息
            data['名称'] = name
            desc = basic_info.get('desc', [])
            data['简介'] = '\n'.join(desc).replace('\r\n\t', '').replace('\r\n\n\n', '').replace(' ', '').replace(
                '\r\n', '\n')

            category = basic_info.get('category', [])
            data['所属类别'] = category
            cates += category

            inspect = item.get('inspect_info', [])
            inspects += inspect

            attributes = basic_info.get('attributes', [])

            # 成因及预防
            data['预防措施'] = item.get('prevent_info', '')
            data['成因'] = item.get('cause_info', '')

            # 症状
            symptom_list = item.get("symptom_info", [[]])[0]
            data['症状'] = list(set([i for i in symptom_list if i and i[0] not in self.stop_words]))

            # 属性对
            for attr in attributes:
                attr_pair = attr.split('：', 1)  # 最多分割一次
                if len(attr_pair) == 2:
                    key = attr_pair[0].strip()
                    value = attr_pair[1].strip()
                    data[key] = value

            # 检查项目
            inspects = item.get('inspect_info', [])
            jcs = []
            for inspect in inspects:
                jc_name = self.get_inspect(inspect)
                if jc_name:
                    jcs.append(jc_name)
            data['检查'] = jcs

            # 食物
            food_info = item.get('food_info', {})
            if food_info:
                data['宜食'] = food_info.get('good', [])
                data['忌食'] = food_info.get('bad', [])
                data['推荐'] = food_info.get('recommand', [])
            else:
                data['宜食'] = []
                data['忌食'] = []
                data['推荐'] = []

            # 药品
            drug_info = item.get('drug_info', [])
            data['药品推荐'] = list(set([i.split('(')[-1].replace(')', '') for i in drug_info if '(' in i]))
            data['药品明细'] = drug_info

            # 转换为英文字段并清洗
            data_modify = {}
            for attr, value in data.items():
                attr_en = self.key_dict.get(attr)
                if not attr_en:
                    continue  # 忽略没有映射的字段

                # 特殊字段清洗
                if attr_en in ['yibao_status', 'get_prob', 'easy_get', 'get_way', "cure_lasttime", "cured_prob"]:
                    data_modify[attr_en] = str(value).replace(' ', '').replace('\t', '')
                elif attr_en in ['cure_department', 'cure_way', 'common_drug']:
                    if isinstance(value, str):
                        data_modify[attr_en] = [i for i in value.split(' ') if i]
                    else:
                        data_modify[attr_en] = value if isinstance(value, list) else []
                elif attr_en == 'acompany':
                    # 使用最大双向分词处理并发症
                    raw = value if isinstance(value, str) else ''.join(value)
                    acompany = [i for i in self.cuter.max_biward_cut(raw) if len(i) > 1]
                    data_modify[attr_en] = acompany
                else:
                    data_modify[attr_en] = value

            # 存入 MongoDB（保留存储逻辑）
            try:
                # 注意：这里仍然使用 MongoDB 存结果，如果你也不想存 MongoDB，请改成写入 JSON 或其他方式
                pymongo.MongoClient()['medical']['medical'].insert_one(data_modify)
                count += 1
                print(count)
            except Exception as e:
                print(f"插入失败: {e}")

        print(f"共处理 {count} 条记录")
        return

    def get_inspect(self, url):
        """
        这个函数原来查 MongoDB 的 jc 集合
        如果你也想从文件读取 jc 数据，可以也改成 JSON 加载
        """
        # 示例：从另一个 JSON 文件加载检查项
        try:
            cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
            jc_path = os.path.join(cur_dir, 'jc_data.json')
            with open(jc_path, 'r', encoding='utf-8') as f:
                jc_data = json.load(f)
            # 假设 jc_data 是列表，每个元素有 'url' 和 'name'
            for item in jc_data:
                if item['url'] == url:
                    return item['name']
            return ''
        except:
            return ''

    def modify_jc(self):
        """
        如果你还想更新 jc 数据，也可以从 JSON 文件读取并处理
        但 MongoDB 的 update 操作需要调整或移除
        """
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        jc_path = os.path.join(cur_dir, 'jc_data.json')
        output = []
        with open(jc_path, 'r', encoding='utf-8') as f:
            jc_data = json.load(f)

        for item in jc_data:
            url = item['url']
            content = item['html']
            selector = etree.HTML(content)
            try:
                name = selector.xpath('//title/text()')[0].split('结果分析')[0]
            except:
                name = item.get('name', url)
            try:
                desc = selector.xpath('//meta[@name="description"]/@content')[0].replace('\r\n\t', '')
            except:
                desc = ''

            # 更新内存中的数据
            item['name'] = name
            item['desc'] = desc
            output.append(item)

        # 写回文件（可选）
        with open(os.path.join(cur_dir, 'jc_data_cleaned.json'), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("检查项数据清洗完成，已保存到 jc_data_cleaned.json")


if __name__ == '__main__':
    handler = MedicalGraph()
    handler.collect_medical()  # 开始处理