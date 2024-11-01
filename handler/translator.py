import json
import time
import requests
import streamlit as st


@st.cache_data(show_spinner=False) 
def hcmus_translate(text):
    url = 'https://api.clc.hcmus.edu.vn/nom_translation/90/1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.request('POST', url, headers=headers, data={'nom_text': text})
    time.sleep(0.1)     
    
    try:
        result = json.loads(response.text)['sentences']
        result = result[0][0]['pair']['modern_text']
        return result
    except:
        print(f'[ERR] "{text}": {response.text}')
        return 'Cannot translate this text.'


@st.cache_data(show_spinner=False)
def hvdic_translate(text):
    def is_nom_text(result):
        for phonetics_dict in result:
            if phonetics_dict['t'] == 3 and len(phonetics_dict['o']) <= 0: 
                return True
        return False
        
    url = 'https://hvdic.thivien.net/transcript-query.json.php'
    headers = { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }
    
    # Request phonetics for Hán Việt (lang=1) first, if the response result is not
    # Hán Việt (contains blank lists) => Request phonetics for Nôm (lang=3)
    for lang in [1, 3]: 
        payload = f'mode=trans&lang={lang}&input={text}'
        response = requests.request('POST', url, headers=headers, data=payload.encode())
        time.sleep(0.1)     
        
        try: 
            result = json.loads(response.text)['result'] 
        except: 
            print(f'[ERR] {text}: {response.text}')
            result = {}
        if not is_nom_text(result): break
    return result


@st.cache_data(show_spinner=False)
def hvdic_render(text):
    phonetics = ''
    for d in hvdic_translate(text):
        if d['t'] == 3 and len(d['o']) > 0: 
            if len(d['o']) == 1: phonetics += d['o'][0] + ' '
            else: phonetics += f'''
                <select name="{d['o'][0]}">
                    {''.join([f'<option><p>{o}</p></option>' for o in d['o']])}
                </select>
            '''.replace('\n', '')
        else: phonetics += '[UNK] '
        
    if len(phonetics) > 0: return phonetics.strip()  
    return 'No response from hvdic => You can clear caches (Press C) and reload.'
    