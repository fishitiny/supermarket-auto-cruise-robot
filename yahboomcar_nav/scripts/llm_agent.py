import re
import requests

# 配置信息
DEEPSEEK_API_KEY = "sk-686564904b2740469e4c5ab566232099"  # 替换为你的API密钥
API_URL = "https://api.deepseek.com/v1/chat/completions"  # 确认API地址正确


'''
你是一个商品分类助手，根据用户描述返回对应的商品序号。商品列表如下：
0: 牛奶
1: 拖鞋
2: 洗衣粉
3: 青菜
4: 钢笔

请按以下规则响应：
1. 直接返回对应的数字（0-4）
2. 没有匹配商品时返回-1
3. 不要包含任何额外内容

示例：
用户：买箱牛奶
回答：0
用户：运动鞋
回答：-1
用户：洗衣服用的
回答：2
用户：绿色蔬菜
回答：3
'''
# 系统提示词
system_prompt = """
You are a product classification assistant. Based on the user's description, return the corresponding product number. The product list is as follows:
0: Milk
1: Slippers
2: Detergent
3: Green vegetables
4: Pen

Please respond according to the following rules:
1. Directly return the corresponding number (0-4)
2. Return -1 if there is no matching product
3. Do not include any extra content

Examples:
User: Buy a box of milk
Answer: 0
User: Sports shoes
Answer: -1
User: For washing clothes
Answer: 2
User: Green vegetables
Answer: 3
"""

def parse_response(text):
    """解析API返回文本中的有效数字"""
    matches = re.findall(r'-?\d+', text)
    for match in matches:
        num = int(match)
        if num in {-1, 0, 1, 2, 3, 4}:
            return num
    return -1

def get_product_number(description):
    """调用DeepSeek API获取商品编号"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description}
        ],
        "temperature": 0,
        "max_tokens": 1  # 限制输出长度
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content'].strip()
            return parse_response(reply)
        print(f"API请求失败，状态码：{response.status_code}")
        return None
    except Exception as e:
        print(f"请求发生错误：{str(e)}")
        return None

def main():
    """主交互循环"""
    while True:
        # 获取用户输入
        user_input = input("\n请输入商品描述（输入 q 退出）: ").strip()
        if user_input.lower() == 'q':
            break
        
        # 获取并验证结果
        result = None
        while result is None:
            result = get_product_number(user_input)
            if result is None:
                retry = input("查询失败，是否重试？(y/n): ").lower()
                if retry != 'y':
                    break
        
        if result is None:
            continue  # 跳过后续处理
            
        print(f"AI回复的商品编号：{result}")
        
        # 用户反馈验证
        while True:
            feedback = input("是否满意该结果？(y 满意/n 重新输入/q 退出): ").lower()
            if feedback == 'y':
                return  # 结束整个程序
            elif feedback == 'n':
                break  # 跳出反馈循环，重新输入描述
            elif feedback == 'q':
                return
            else:
                print("无效输入，请重新选择。")
        
        # 如果不满意则继续循环

if __name__ == "__main__":
    print("商品查询系统启动（输入 q 退出）")
    main()
    print("系统已退出")