import urllib.request
import urllib.parse
from lxml import etree
import json
import time
import os

'''
疾病信息爬虫（无数据库版）
功能：采集疾病详情并保存为 JSON 文件
'''


class DiseaseSpider:
    def __init__(self,disease_file='diseases.json',inspect_file='inspects.json'):
        '''
        disease_file:疾病相关描述的json文件保存路径
        inspect_file:医学检查项相关描述的json文件保存路径
        '''
        # 保存文件路径
        self.disease_file =disease_file
        self.inspect_file =inspect_file

    '''根据 URL 获取 HTML 内容'''

    def get_html(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/51.0.2704.63 Safari/537.36'
        }
        req = urllib.request.Request(url=url, headers=headers)
        try:
            res = urllib.request.urlopen(req, timeout=10)
            html = res.read()
            # 尝试用 utf-8 解码，失败则用 gbk
            try:
                return html.decode('utf-8')
            except UnicodeDecodeError:
                return html.decode('gbk')
        except Exception as e:
            print(f"获取页面失败: {url}, 错误: {e}")
            return ''

    '''主函数：爬取疾病数据并保存为 JSON'''

    def spider_main(self, start_page=1, end_page=100):
        print(f"开始爬取疾病数据（{start_page} ~ {end_page}）...")
        all_data = []

        # 如果文件已存在，加载已有数据（实现增量采集）
        if os.path.exists(self.disease_file):
            with open(self.disease_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            print(f"已加载 {len(all_data)} 条历史数据")

        existing_ids = {item['page_id'] for item in all_data}  # 避免重复采集

        for page in range(start_page, end_page + 1):
            if page in existing_ids:
                print(f"[跳过] 第 {page} 页已存在")
                continue

            try:
                basic_url = f'http://jib.xywy.com/il_sii/gaishu/{page}.htm'
                cause_url = f'http://jib.xywy.com/il_sii/cause/{page}.htm'
                prevent_url = f'http://jib.xywy.com/il_sii/prevent/{page}.htm'
                symptom_url = f'http://jib.xywy.com/il_sii/symptom/{page}.htm'
                inspect_url = f'http://jib.xywy.com/il_sii/inspect/{page}.htm'
                treat_url = f'http://jib.xywy.com/il_sii/treat/{page}.htm'
                food_url = f'http://jib.xywy.com/il_sii/food/{page}.htm'
                drug_url = f'http://jib.xywy.com/il_sii/drug/{page}.htm'

                data = {
                    'page_id': page,
                    'url': basic_url,
                    'basic_info': self.basicinfo_spider(basic_url),
                    'cause_info': self.common_spider(cause_url),
                    'prevent_info': self.common_spider(prevent_url),
                    'symptom_info': self.symptom_spider(symptom_url),
                    'inspect_info': self.inspect_spider(inspect_url),
                    'treat_info': self.treat_spider(treat_url),
                    'food_info': self.food_spider(food_url),
                    'drug_info': self.drug_spider(drug_url)
                }

                # 只有成功采集才添加
                if data['basic_info']:  # 基本信息存在说明页面有效
                    all_data.append(data)
                    print(f"[成功] 爬取第 {page} 页: {data['basic_info']['name']}")
                else:
                    print(f"[空页] 第 {page} 页无数据，跳过")

            except Exception as e:
                print(f"[失败] 爬取第 {page} 页失败: {e}")

            # 防封：每爬一页暂停 1 秒
            time.sleep(1)

        # 保存到 JSON 文件
        with open(self.disease_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 疾病数据已保存到 {self.disease_file}，共 {len(all_data)} 条")

    '''基本信息解析'''

    def basicinfo_spider(self, url):
        html = self.get_html(url)
        if not html:
            return {}
        selector = etree.HTML(html)
        try:
            title = selector.xpath('//title/text()')[0]
            category = selector.xpath('//div[@class="wrap mt10 nav-bar"]/a/text()')
            desc_list = selector.xpath('//div[@class="jib-articl-con jib-lh-articl"]/p/text()')
            ps = selector.xpath('//div[@class="mt20 articl-know"]/p')
            infobox = []
            for p in ps:
                info = p.xpath('string(.)').replace('\r', '').replace('\n', '').replace('\xa0', '').replace('   ',
                                                                                                            '').replace(
                    '\t', '')
                if info.strip():
                    infobox.append(info.strip())

            return {
                'category': [cat.strip() for cat in category],
                'name': title.replace('的简介', '').strip(),
                'desc': ''.join(desc_list).strip(),
                'attributes': infobox
            }
        except Exception as e:
            print(f"解析 basic_info 失败 {url}: {e}")
            return {}

    '''治疗信息解析'''

    def treat_spider(self, url):
        html = self.get_html(url)
        if not html:
            return []
        selector = etree.HTML(html)
        ps = selector.xpath('//div[starts-with(@class,"mt20 articl-know")]/p')
        infobox = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r', '').replace('\n', '').replace('\xa0', '').replace('   ',
                                                                                                        '').replace(
                '\t', '')
            if info.strip():
                infobox.append(info.strip())
        return infobox

    '''药品推荐解析'''

    def drug_spider(self, url):
        html = self.get_html(url)
        if not html:
            return []
        selector = etree.HTML(html)
        drugs = [name.replace('\n', '').replace('\t', '').replace(' ', '')
                 for name in selector.xpath('//div[@class="fl drug-pic-rec mr30"]/p/a/text()')]
        return drugs

    '''饮食建议解析'''

    def food_spider(self, url):
        html = self.get_html(url)
        if not html:
            return {}
        selector = etree.HTML(html)
        divs = selector.xpath('//div[@class="diet-img clearfix mt20"]')
        try:
            good = [food.strip() for food in divs[0].xpath('./div/p/text()')]
            bad = [food.strip() for food in divs[1].xpath('./div/p/text()')]
            recommand = [food.strip() for food in divs[2].xpath('./div/p/text()')]
            return {'good': good, 'bad': bad, 'recommand': recommand}
        except:
            return {}

    '''症状信息解析'''

    def symptom_spider(self, url):
        html = self.get_html(url)
        if not html:
            return {'symptoms': [], 'symptoms_detail': []}
        selector = etree.HTML(html)
        symptoms = selector.xpath('//a[@class="gre"]/text()')
        ps = selector.xpath('//p')
        detail = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r', '').replace('\n', '').replace('\xa0', '').replace('   ',
                                                                                                        '').replace(
                '\t', '')
            if info.strip():
                detail.append(info.strip())
        return {'symptoms': symptoms, 'symptoms_detail': detail}

    '''检查项目解析'''

    def inspect_spider(self, url):
        html = self.get_html(url)
        if not html:
            return []
        selector = etree.HTML(html)
        inspects = selector.xpath('//li[@class="check-item"]/a/@href')
        return inspects

    '''通用文本解析模块'''

    def common_spider(self, url):
        html = self.get_html(url)
        if not html:
            return ''
        selector = etree.HTML(html)
        ps = selector.xpath('//p')
        texts = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r', '').replace('\n', '').replace('\xa0', '').replace('   ',
                                                                                                        '').replace(
                '\t', '')
            if info.strip():
                texts.append(info.strip())
        return '\n'.join(texts)

    '''检查项页面 HTML 抓取（保存为 JSON）'''

    def inspect_crawl(self, start_page=1, end_page=3684):
        print(f"开始爬取检查项页面 HTML（{start_page} ~ {end_page}）...")
        all_inspects = []

        if os.path.exists(self.inspect_file):
            with open(self.inspect_file, 'r', encoding='utf-8') as f:
                all_inspects = json.load(f)
            print(f"已加载 {len(all_inspects)} 条检查项数据")

        existing_ids = {item['page_id'] for item in all_inspects}

        for page in range(start_page, end_page + 1):
            if page in existing_ids:
                print(f"[跳过] 检查项第 {page} 页已存在")
                continue

            try:
                url = f'http://jck.xywy.com/jc_{page}.html'
                html = self.get_html(url)
                if html:
                    data = {
                        'page_id': page,
                        'url': url,
                        'html': html  # 可改为只存关键部分以节省空间
                    }
                    all_inspects.append(data)
                    print(f"[成功] 爬取检查项第 {page} 页")
                else:
                    print(f"[失败] 获取检查项第 {page} 页失败")
            except Exception as e:
                print(f"[异常] 检查项第 {page} 页: {e}")

            time.sleep(1)  # 防封

        # 保存
        with open(self.inspect_file, 'w', encoding='utf-8') as f:
            json.dump(all_inspects, f, ensure_ascii=False, indent=2)
        print(f"✅ 检查项数据已保存到 {self.inspect_file}")


# ========================
#  🚀 运行爬虫
# ========================
if __name__ == '__main__':
    spider = DiseaseSpider(disease_file='diseases.json',
                           inspect_file='inspects.json')
    # 选择运行一个任务：
    # 1. 爬疾病数据（建议先试 1-10）
    spider.spider_main(start_page=1, end_page=10) #  end_page=11000

    # 2. 爬医学检查项目 HTML（可选）
    # spider.inspect_crawl(start_page=1, end_page=10)