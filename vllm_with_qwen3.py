import json
import os
import traceback
import time
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel


class NERInfo(BaseModel):
    姓名: List[str]
    联系方式: List[str]
    地址: List[str]
    工作单位: List[str]
    职位: List[str]
    邮箱: List[str]
    所属部门: List[str]
    职称: List[str]
    传真: List[str]
    个人主页: List[str]
    社交信息: List[str]


def content_ner_with_llm(content):
    llm_host = os.environ.get('llm_host')
    if not llm_host:
        raise Exception('llm_host is not set')
    prompt_str = "你是一个命名实体识别（NER）助手，需要从给定文本中提取以下实体信息：（姓名、联系方式、地址、工作单位、职位、邮箱、所属部门、职称、传真、个人主页、社交信息）。输出结果为JSON格式，其中：\n1.键为识别出的信息类别值为对应的内容\n2.如果未识别出来则json不需要包含这个key\n3.如果识别出多个值，使用列表展示（仅包含值，不嵌套）。示例 :{'联系方式': ['123456', '789012'],\"姓名\":[\"sam\",\"wang\"]}\n\n\n你的输出必须只是格式良好的 JSON 格式字符串。输出应该是单个可以被 json.loads 解析的 JSON 对象\n\n• 不要包含警告、注释或任何额外信息；只输出要求的部分\n\n• 不要重复想法、引用、事实或资源\n\n• 不要用相同的开头词语开始条目\n\n• 确保 JSON 对象中的所有键都使用双引号\n\n• 使用反斜杠 \\ 转义特殊字符\n\n• 确保 JSON 对象中的所有值都正确格式化为字符串、数字、数组或对象\n\n• 确保 JSON 对象中没有多余的逗号\n\n• 确保创建输出时遵循所有这些指令 内容如下："
    if content:
        model_name = "/root/.cache/huggingface/Qwen3-8B"
        result = request_vllm_model_with_openai_client(llm_host, prompt_str + content, model_name,
                                                       schema=NERInfo.model_json_schema())
    return None


def request_vllm_model_with_openai_client(host_llm, prompt, model_name, image_path=None, schema=None) -> Optional[str]:
    time_now = time.time()
    if not host_llm:
        host_llm = os.environ.get('llm_host')
        if not host_llm:
            raise Exception('llm_host is not set')
    try:
        client = OpenAI(base_url=host_llm, api_key="none")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]

            }
        ]
        # 多模态模型
        if image_path:
            pass
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.8,
            top_p=0.9,
            max_tokens=8192,
            presence_penalty=1.3,
            extra_body={"top_k": 20, "guided_json": schema, "chat_template_kwargs": {"enable_thinking": False}},
        )
        print("请求耗时:", time.time() - time_now, "请求长度:", len(prompt))
        print("请求参数:\n", prompt, "\n")
        print("请求结果:\n", response.choices[0].message.content, "\n")
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(traceback.format_exc())
    return None


if __name__ == '__main__':
    os.environ["llm_host"] = "http://192.168.205.142:8011/v1/"
    content_ner_with_llm(
        "我样样事总喜欢革新；我虽然没做过国民党党员，可是我们一家人和革命的关系很深。我后来到了够岁数的时候还加入了国民党前身的同盟会(不是我自己加入的，是林贯虹弟兄在日本给我加入的)。我祖父根本就不是大清帝国的一个忠实的老百姓，所以他一生不愿考科举或做官，以后出洋回来保举也不愿接受。他除了因为感情生活上失望以外(参看第十六章)，他的革命思想也是他出洋回来不做官的一个理由。有一年夏天的晚上我差不多七岁的时候，祖父和父亲站在槅子门边谈英国的宪法和人权的事。我一点不懂，惟说到人民有权选举等等事，我觉得非常有意思。(其实我也不知是什么，不过我一小就觉得什么事可以由我做点主总是好的。只要别人叫我做什么，我总问为什么你要我做这个、做那个呢？)我就在旁边问什么叫人民有权？权是什么？父亲回我，又多嘴了，没有规矩！说完了笑笑。因为那时中国家庭规矩长辈说话，小孩子们不能插嘴的，不管是非好奇也不能问的，须等说完以后，才可以小声问一下。若是长辈不愿解说就完了，也不能再追问。但是我的祖父和父亲非常讲新法的人，我本人自己又是一小惯的不得了，所以问时父亲常解说给我听，祖父更喜欢人多问，所以养成我“打破沙锅璺到底，还问沙锅怎么起”的习惯来了。父亲骂我多嘴是照规矩，笑笑是表示以后再告诉我，免我失望，这样子把我越惯越没规矩。所以我到今天也学不会外国开会式的交际谈话，非得等一个人一串话说完了你才能说，等轮到我说时我早把我要说的话忘记了。并且碰到个贫嘴的人你不打他的岔怎么止得住他呢？")
