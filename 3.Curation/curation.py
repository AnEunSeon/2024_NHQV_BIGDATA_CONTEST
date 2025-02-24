#terminal에서 실행
#pip install openai==0.28
#pip install openai azure-storage-blob python-dotenv pandas
#pip install pandas
#pip install python-dotenv

import openai
import os
import pandas as pd
import io
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import re
import time
from datetime import datetime
import requests

#env 파일에서 환경 변수 로드
load_dotenv()
#Azure OpenAI API 설정
openai.api_type = "azure"
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')  
openai.api_version = "2024-05-01-preview"
#Azure Storage Blob 서비스 설정
blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))

#종목 선정에 필요한 ETF 점수 데이터 불러오기
def load_dataset_from_blob(container_name, blob_name):
    # Blob 클라이언트 생성
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    stream = io.BytesIO(blob_client.download_blob().readall())
    return pd.read_csv(stream)

#사용자군에 따른 선호도 계산
def calculate_combined_value(container_name, blob_name1, blob_name2, user_choices):
    dataset1 = load_dataset_from_blob(container_name, blob_name1)
    dataset2 = load_dataset_from_blob(container_name, blob_name2)
    user_df = dataset1[['cluster'] + user_choices].copy(deep=True)
    user_df['rank'] = user_df[user_choices].sum(axis=1)
    df = user_df[['cluster', 'rank']]
    res_df = dataset2.copy()
    res_df.drop('tck_iem_cd',axis=1,inplace=True)
    res_df=res_df.groupby('cluster').mean().reset_index()
    norm_df=res_df.drop('cluster',axis=1)
    row_min =norm_df.min(axis=0)
    norm_df =(norm_df.subtract(row_min, axis=1))
    res_df=pd.concat([res_df['cluster'],norm_df],axis=1)
    for i in range(len(res_df)):
        cluster=res_df.loc[i,'cluster']
        rank_value = df.loc[df['cluster'] == cluster, 'rank'].values[0]
        res_df.iloc[i, 1:] *= rank_value

    profit_score = res_df['profit'].sum(axis=0)
    variation_score = res_df['variation'].sum(axis=0)
    volume_score = res_df['volume'].sum(axis=0)
    total_score = profit_score + variation_score + volume_score

    profit_score = round(profit_score / total_score * 100)
    variation_score = round(variation_score / total_score * 100)
    volume_score = round(volume_score / total_score * 100)
    ratio_df=pd.DataFrame({'profit':[profit_score],'variation': [variation_score],'volume': [volume_score]})
    return ratio_df

#선호도에 따른 ETF 추천 점수 산출 및 추천 ETF 종목 선정
def etf_score(container_name, blob_name3, profit_val, variation_val, volume_val):
    dataset3 = load_dataset_from_blob(container_name, blob_name3)
    etf_info=dataset3.copy(deep=True)
    etf_rec_sc=pd.DataFrame(columns=['profit','variation','volume','etf_score'])
    etf_rec_sc['profit'] = etf_info['acl_pft_rt_z_sor'] * (profit_val/100)
    etf_rec_sc['variation'] = etf_info['vty_z_sor'] * (variation_val/100)
    etf_rec_sc['volume'] = etf_info['volume'] * (volume_val/100)
    etf_rec_sc['etf_score'] = etf_rec_sc.sum(axis=1)
    etf_rec_sc=pd.concat([etf_info['etf_iem_cd'],etf_rec_sc],axis=1)
    etf_rec_sc=etf_rec_sc.sort_values('etf_score', ascending=False)
    curate_etf = list(etf_rec_sc[0:10]['etf_iem_cd'])
    return curate_etf

#ETF 특징 추출 함수
def etf_description_response(etf_code):
    descriptions = {}
    prompt = f"ETF {etf_code}를 다른 미국 ETF 종목과 차별화되는 3가지 주요 특징을 명사 키워트 형태로 추출합니다."
    try:
        response = openai.ChatCompletion.create(
            deployment_id="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "당신은 금융 분야의 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        extracted_text = response["choices"][0]["message"]["content"].strip()
        descriptions[etf_code] = [desc.strip("1234567890. ") for desc in extracted_text.split("\n") if desc.strip()]
    except Exception as e:
        descriptions[etf_code] = ["특징 추출 실패"]
        print(f"ETF {etf_code} 설명 생성 중 오류 발생: {e}")
    return descriptions

#ETF 특징 출력문 형태 변환 함수
def print_etf_descriptions(etf_descriptions):
    if not etf_descriptions:
        print("ETF 설명이 없습니다.")
        return
    for etf, descriptions in etf_descriptions.items():
        print()
        print(etf)
        for i, desc in enumerate(descriptions, start=1):
            print(f"{i}. {desc}")
        print()

#챗봇 대화 플로우 설계
def chatbot_conversation(container_name, blob_name1, blob_name2, blob_name3):
    print("안녕하세요. 미국 ETF 큐레이션 서비스를 시작할게요.\n고객님의 정보를 바탕으로 맞춤형 ETF를 살펴보세요! 해당 서비스를 위해 고객님의 정보를 알려주시겠어요?")
    user_choices = []
    time.sleep(2)
    #고객 나이 입력
    age_prompt = "10-20대 / 30대 / 40대 / 50대 / 60대 이상 중 고객님의 연령대를 입력해주세요."
    age_options = {"10-20대": '21', "30대": '22', "40대": '23', "50대": '24', "60대 이상": '25'}
    print(age_prompt)
    while True:
        age_response = input(":")
        if age_response in age_options:
            user_choices.append(age_options.get(age_response, None))
            break
        else:
            print("1020대 / 30대 / 40대 / 50대 / 60대 이상 중에서 입력해주세요.")
    #투자 규모 입력
    amount_prompt = "3천만원 미만 / 3천만원 이상 / 1억원 이상 / 10억원 이상 중 고객님의 투자 규모를 입력해주세요."
    amount_options = {"3천만원 미만": '31', "3천만원 이상": '32', "1억원 이상": '33', "10억원 이상": '34'}
    print(amount_prompt)
    while True:
        amount_response = input(":")
        if amount_response in amount_options:
            user_choices.append(amount_options.get(amount_response, None))
            break
        else:
            print("3천만원 미만 / 3천만원 이상 / 1억원 이상 / 10억원 이상 중에서 입력해주세요.")

    #투자 실력 선택
    skill_prompt = "투자 고수 / 일반 투자자 중 고객님의 투자 실력을 입력해주세요."
    skill_options = {"투자 고수": '11', "일반 투자자": '12'}
    print(skill_prompt)
    while True:
        skill_response = input(":")
        if skill_response in skill_options:
            user_choices.append(skill_options.get(skill_response, None))
            break
        else:
            print("투자 고수 / 일반 투자자 중에서 입력해주세요.")

    #투자 스타일 분석
    combined_value = calculate_combined_value(container_name, blob_name1, blob_name2, user_choices)
    profit = combined_value['profit'].iloc[0]
    variation = combined_value['variation'].iloc[0]
    volume = combined_value['volume'].iloc[0]
    print(f"고객님과 유사한 고객님들의 투자 스타일은 수익성 {profit:.0f}% 안정성 {variation:.0f}% 대중성 {volume:.0f}%을 중시해요!")
    
    # 투자 스타일 비중 조정
    time.sleep(2)
    ratio_prompt="해당 비중으로 미국 ETF 종목 추천을 진행할까요? 이대로 진행하길 바라시면 네, 비중 조정을 바라면 아니오를 입력해주세요."
    print(ratio_prompt)
    while True:
        ratio_response = input(":")
        if ratio_response == "아니오":
            print("비중 조정을 시작할게요. 총합이 100이 되도록 입력해주세요. %는 생략하고 숫자만 입력해주세요.")
            time.sleep(2)
            while True:
                profit_prompt="원하시는 수익성 비중을 입력해주세요."
                print(profit_prompt)
                profit = int(input(":"))
                variation_prompt="원하시는 안정성 비중을 입력해주세요."
                print(variation_prompt)
                variation = int(input(":"))
                volume_prompt="원하시는 대중성 비중을 입력해주세요."
                print(volume_prompt)
                volume = int(input(":"))
                if profit+variation+volume!=100:
                    print("총합이 100이 되지 않아요. 총합이 100이 되도록 다시 입력해주세요.")
                    time.sleep(2)
                else:
                    print(f"고객님의 투자 스타일은 수익성 {profit:.0f}% 안정성 {variation:.0f}% 대중성 {volume:.0f}%을 중시해요!")
                    break
            break
        elif ratio_response == "네":
            break
        else:
            print("네 / 아니오 중에서 입력해주세요.")
    time.sleep(1)
    print("고객님의 투자 스타일로 미국 ETF 종목 추천을 진행할게요!")

    #추천 ETF 종목 제시 
    time.sleep(1)
    etf_recom = etf_score(container_name, blob_name3, profit, variation, volume)
    print(f"고객님에게 추천드리는 ETF는 {', '.join(etf_recom)}이에요.\n궁금하신 ETF를 입력해주세요. 입력하신 ETF의 주요 특징을 알려드릴게요!\n궁금하신 ETF가 없으시면 아니오를 입력해주세요.")
    while True:
        etf_select = input(":").strip()
        etf_select_lower = etf_select.lower()  
        if etf_select_lower == "아니오":
            print("미국 ETF 큐레이션 서비스를 종료할게요.")
            break
        etf_recom_lower = [etf.lower() for etf in etf_recom]
        if etf_select_lower in etf_recom_lower:
            etf_original = next(etf for etf in etf_recom if etf.lower() == etf_select_lower)
            etf_describ = etf_description_response(etf_original)
            print_etf_descriptions(etf_describ)
            time.sleep(1)
            print("다른 ETF도 궁금하신가요?\n궁금하신 ETF가 있으시면 ETF를 입력해주세요.\n더 이상 궁금한 ETF가 없으시면 '아니오'를 입력해주세요.")
        else:
            print("궁금하신 ETF를 정확하게 입력해주세요.")
            
# 실행
if __name__ == "__main__":
    chatbot_conversation("dataset", "cust_rk.csv", "stk_sc.csv","fin_etf.csv")