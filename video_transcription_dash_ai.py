# !pip install dash dash-bootstrap-components openai-whisper yt-dlp torch nltk requests google.adk
import nltk
from nltk.util import ngrams
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import date
import os
import requests
import textwrap
import time
import torch
import warnings
import whisper
import yt_dlp
from google.colab import userdata
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from IPython.display import display, Markdown, HTML
from collections import Counter
import re

# Baixar recursos do NLTK
nltk.download('stopwords')

# Define a precisão correta para FP32 na CPU
torch.set_default_dtype(torch.float32)

# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Função para extrair palavra-chave do resumo
def extrair_termo_principal(texto, idioma="pt", n_gram=2):
    stop_words = {
        "pt": ["a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
               "para", "com", "que", "e", "é", "um", "uma", "uns", "umas", "se", "por", "sobre"],
        "en": ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
               "by", "from", "up", "about", "into", "over", "after"],
        "es": ["el", "la", "los", "las", "de", "del", "en", "y", "o", "que", "es", "un", "una",
               "unos", "unas", "para", "con", "por", "sobre"]
    }.get(idioma, [])
    
    texto_limpo = re.sub(r'[^\w\s]', '', texto.lower())
    palavras = texto_limpo.split()
    palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 2]
    
    # Gerar n-grams
    n_grams = [' '.join(gram) for gram in ngrams(palavras_filtradas, n_gram)]
    contagem = Counter(n_grams)
    
    return contagem.most_common(1)[0][0] if contagem else ""

# Função para extrair ID do vídeo do URL do YouTube
def extrair_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Função para baixar o áudio do YouTube usando yt-dlp
def baixar_audio(url, update_callback=None):
    video_id = extrair_video_id(url)
    if not video_id:
        print("Erro: Não foi possível extrair o ID do vídeo da URL")
        return None
    
    nome_arquivo = f"{video_id}.wav"
    
    # Verifica se o arquivo já existe
    if os.path.exists(nome_arquivo):
        print(f"Arquivo {nome_arquivo} já existe. Usando o arquivo existente.")
        return nome_arquivo

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': video_id,  # Usa o ID do vídeo como nome base
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'progress_hooks': [],
        'overwrite': True,
    }
    
    def progress_hook(d):
        if update_callback:
            if d['status'] == 'downloading':
                if d.get('total_bytes_estimate'):
                    percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 50
                    update_callback(percent)
                elif d.get('total_bytes'):
                    percent = (d['downloaded_bytes'] / d['total_bytes']) * 50
                    update_callback(percent)
    
    ydl_opts['progress_hooks'].append(progress_hook)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Verifica se o arquivo foi criado com a extensão correta
        if os.path.exists(nome_arquivo):
            return nome_arquivo
        else:
            print(f"Erro: Arquivo {nome_arquivo} não foi criado")
            return None
    except Exception as e:
        print(f"Erro ao baixar o áudio: {e}")
        return None

# Função para transcrever o áudio usando Whisper
def transcrever_audio(nome_arquivo, idioma, update_callback=None):
    try:
        modelo = whisper.load_model("base", device="cpu")
        resultado = modelo.transcribe(nome_arquivo, language=idioma, temperature=0, word_timestamps=True)
        transcricao_formatada = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in resultado["segments"]
        ]
        if update_callback:
            update_callback(100)
        return transcricao_formatada
    except Exception as e:
        print(f"Erro ao transcrever o áudio: {e}")
        return ["Erro na transcrição."]

# Função para exportar transcrição em formato SRT
def exportar_srt(transcricao, nome_arquivo="legendas.srt"):
    try:
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            for i, seg in enumerate(transcricao, 1):
                start = float(seg["start"])
                end = float(seg["end"])
                text = seg["text"]
                f.write(f"{i}\n")
                f.write(f"{start//3600:02d}:{(start%3600)//60:02d}:{start%60:06.3f} --> {end//3600:02d}:{(end%3600)//60:02d}:{end%60:06.3f}\n")
                f.write(f"{text}\n\n")
        return nome_arquivo
    except Exception as e:
        print(f"Erro ao exportar SRT: {e}")
        return None

# Configura o cliente da SDK do Gemini
def config_ai():
    try:
        api_key = userdata.get('GOOGLE_API_KEY')
        if not api_key:
            print("Erro:4: GOOGLE_API_KEY não encontrada no userdata.")
            return None, None
        os.environ["GOOGLE_API_KEY"] = api_key
        client = genai.Client()
        MODEL_ID = "gemini-2.0-flash"
        return client, MODEL_ID
    except Exception as e:
        print(f"Erro ao configurar a API do Gemini: {str(e)}")
        return None, None

# Função auxiliar para chamar agentes
def call_agent(name, description, topico, subject, instrucao, model_id, client, tools=False):
    try:
        agent = Agent(name=name, model=model_id, description=description, instruction=instrucao)
        agent.tools = [google_search] if tools else []
        message_text = f"Tópico: {topico}-{subject}"
        session_service = InMemorySessionService()
        session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
        runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
        content = types.Content(role="user", parts=[types.Part(text=message_text)])
        final_response = ""
        for event in runner.run(user_id="user1", session_id="session1", new_message=content):
            if event.is_final_response():
                for part in event.content.parts:
                    if part.text is not None:
                        final_response += part.text + "\n"
        if not final_response.strip():
            print(f"Agente {name} retornou resposta vazia para o tópico: {topico}")
        return final_response
    except Exception as e:
        print(f"Erro no agente {name}: {str(e)}")
        return ""

# Agente Resumidor
def agente_resumidor(texto, model_id, client):
    name = "agente_resumidor"
    description = "Agente que gera resumos concisos de textos."
    instruction = """
    Gere um resumo conciso do texto fornecido, destacando os pontos principais em até 100 palavras.
    """
    subject = f"Texto a ser resumido: {texto}"
    resumo = call_agent(name, description, texto, subject, instruction, model_id, client, tools=False)
    return resumo

# Agente Analisador de Sentimentos
def agente_analisador_sentimentos(texto, model_id, client):
    name = "agente_analisador_sentimentos"
    description = "Agente que analisa o sentimento geral de um texto."
    instruction = """
    Analise o sentimento geral do texto fornecido e classifique-o como 'positivo', 'negativo' ou 'neutro'.
    Forneça apenas a classificação do sentimento.
    """
    subject = f"Texto a ser analisado: {texto}"
    sentimento = call_agent(name, description, texto, subject, instruction, model_id, client, tools=False)
    return sentimento

# Agente Buscador
def agente_buscador(topico, data_de_hoje, model_id, client):
    name = "agente_buscador"
    description = "Agente que busca informações no Google."
    tools = True
    instruction = """
    Você é um assistente de pesquisa. Use a ferramenta de busca do Google (google_search) para recuperar as últimas notícias de lançamentos muito relevantes sobre o tópico abaixo. Foque em no máximo 5 lançamentos relevantes, com base na quantidade e entusiasmo das notícias. Lançamentos devem ser atuais, de no máximo um mês antes da data de hoje.
    """
    subject = f"Tópico: {topico}\nData de hoje: {data_de_hoje}"
    lancamento = call_agent(name, description, topico, subject, instruction, model_id, client, tools)
    return lancamento

# Agente Planejador
def agente_planejador(topico, lancamentos_buscados, model_id, client):
    name = "agente_planejador"
    description = "Agente que planeja posts."
    tools = True
    instruction = """
    Você é um planejador de conteúdo especialista em redes sociais. Com base na lista de lançamentos fornecida, use a ferramenta de busca do Google para criar um plano sobre os pontos mais relevantes a abordar em um post para cada lançamento. Escolha o tema mais relevante e retorne o tema, seus pontos principais e um plano com os assuntos a serem abordados no post.
    """
    subject = f"Tópico: {topico}\nLançamentos buscados: {lancamentos_buscados}"
    plano_do_post = call_agent(name, description, topico, subject, instruction, model_id, client, tools)
    return plano_do_post

# Agente Redator
def agente_redator(topico, plano_de_post, model_id, client):
    name = "agente_redator"
    description = "Agente redator de posts engajadores para Instagram."
    instruction = """
    Você é um Redator Criativo especializado em posts virais para redes sociais. Escreve posts para a Alura, a maior escola online de tecnologia do Brasil. Utilize o plano de post fornecido e escreva um rascunho de post para Instagram. O post deve ser engajador, informativo, com linguagem simples e incluir 2 a 4 hashtags.
    """
    subject = f"Tópico: {topico}\nPlano de post: {plano_de_post}"
    rascunho = call_agent(name, description, topico, subject, instruction, model_id, client, tools=False)
    return rascunho

# Agente Revisor
def agente_revisor(topico, rascunho_gerado, model_id, client):
    name = "agente_revisor"
    description = "Agente revisor de post para redes sociais."
    instruction = """
    Você é um Editor e Revisor de Conteúdo meticuloso, especializado em posts para Instagram. Revise o rascunho abaixo, verificando clareza, concisão, correção e tom (adequado para público jovem, 18-30 anos). Se estiver bom, responda 'O rascunho está ótimo e pronto para publicar!'. Caso contrário, aponte problemas e sugira melhorias.
    """
    subject = f"Tópico: {topico}\nRascunho: {rascunho_gerado}"
    texto_revisado = call_agent(name, description, topico, subject, instruction, model_id, client, tools=False)
    return texto_revisado

# Função para executar todos os agentes
def run_agentes(topico, model_id, client):
    data_de_hoje = date.today().strftime("%d/%m/%Y")
    lancamentos_buscados = agente_buscador(topico, data_de_hoje, model_id, client)
    plano_de_post = agente_planejador(topico, lancamentos_buscados, model_id, client)
    rascunho_de_post = agente_redator(topico, plano_de_post, model_id, client)
    post_final = agente_revisor(topico, rascunho_de_post, model_id, client)
    return post_final

# Layout do dashboard
app.layout = dbc.Container([
    html.H1("🎤 Transcrição e Análise de Vídeos do YouTube", className="text-center my-4"),
    dcc.Store(id="transcricao-store", data={"completed": False, "texto": ""}),
    dcc.Store(id="palavra-chave-store", data=""),
    dbc.Row([
        dbc.Col([
            html.H2("Transcrição de Áudio do YouTube", className="mt-4 text-secondary"),
            html.Label("URL do Vídeo do YouTube:", className="fw-bold"),
            dcc.Input(id="input-url", type="text", placeholder="Insira a URL do vídeo do YouTube aqui...", className="form-control mb-2"),
            dcc.Dropdown(
                id="idioma-transcricao",
                options=[
                    {"label": "Inglês", "value": "en"},
                    {"label": "Português", "value": "pt"},
                    {"label": "Espanhol", "value": "es"},
                ],
                value="pt",
                className="mb-2"
            ),
            dbc.Button("Processar Áudio", id="btn-processar-audio", color="primary", className="mt-2 w-100"),
            dbc.Progress(id="barra-progresso", value=0, max=100, color="success", className="mt-2"),
        ], width=6, md=6),
        dbc.Col([
            html.H2("Criação de Posts para Instagram", className="mt-4 text-primary"),
            html.Label("Tópico do Post:", className="fw-bold"),
            dcc.Input(id="input-topico-agente", type="text", placeholder="Digite o tópico aqui...", className="form-control mb-2"),
            dbc.Button("Gerar Post", id="btn-gerar-post", color="success", className="mt-2 w-100"),
        ], width=6, md=6),
    ]),
    dbc.Tabs([
        dbc.Tab(label="Transcrição", children=[
            dcc.Loading(
                id="loading-transcricao",
                type="circle",
                children=html.Div(id="transcricao", children="Aguardando transcrição...", className="p-3 border rounded bg-light")
            )
        ]),
        dbc.Tab(label="Resumo", children=[
            dcc.Loading(
                id="loading-resumo",
                type="circle",
                children=html.Div(id="resumo", className="p-3 border rounded bg-light")
            )
        ]),
        dbc.Tab(label="Sentimentos", children=[
            dcc.Loading(
                id="loading-sentimentos",
                type="circle",
                children=[
                    html.Div(id="sentimentos-texto", className="p-3"),
                    dcc.Graph(id="sentimentos-grafico")
                ]
            )
        ]),
        dbc.Tab(label="Post", children=[
            dcc.Loading(
                id="loading-post",
                type="circle",
                children=html.Div(id="output-agente", className="p-3 border rounded bg-light")
            )
        ])
    ], className="mt-4")
], fluid=True)

# Callback para processar o áudio e atualizar a transcrição
@app.callback(
    [
        Output("transcricao", "children"),
        Output("barra-progresso", "value"),
        Output("transcricao-store", "data"),
    ],
    [Input("btn-processar-audio", "n_clicks")],
    [State("input-url", "value"), State("idioma-transcricao", "value")],
    prevent_initial_call=True
)
def atualizar_transcricao(n_clicks, url_video, idioma):
    print("Callback atualizar_transcricao acionado")
    if not n_clicks or not url_video:
        print("Sem cliques ou URL vazia")
        return "Insira um URL válido e pressione o botão.", 0, {"completed": False, "texto": ""}

    def update_progress(value):
        return value

    arquivo_wav = baixar_audio(url_video, update_callback=update_progress)
    if arquivo_wav:
        transcricao_formatada = transcrever_audio(arquivo_wav, idioma, update_callback=update_progress)
        if isinstance(transcricao_formatada, list) and transcricao_formatada and "Erro" not in transcricao_formatada[0]:
            print(f"Transcrição gerada com {len(transcricao_formatada)} segmentos")
            texto_transcricao = " ".join([seg["text"] for seg in transcricao_formatada])
            return (
                html.Ul([html.Li(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}") for seg in transcricao_formatada]),
                100,
                {"completed": True, "texto": texto_transcricao}
            )
        else:
            print("Erro na transcrição")
            return "Erro ao processar a transcrição.", 0, {"completed": False, "texto": ""}
    else:
        print("Erro ao baixar áudio")
        return "Erro ao baixar o áudio.", 0, {"completed": False, "texto": ""}

# Callback para gerar o resumo e extrair palavra-chave
@app.callback(
    [
        Output("resumo", "children"),
        Output("palavra-chave-store", "data"),
    ],
    [Input("transcricao-store", "data")],
    [State("idioma-transcricao", "value")],
    prevent_initial_call=True
)
def gerar_resumo(transcricao_data, idioma):
    print("Callback gerar_resumo acionado")
    if not transcricao_data or not transcricao_data.get("completed", False):
        print("Transcrição não completa ou inexistente")
        return "Aguardando transcrição...", ""

    texto = transcricao_data.get("texto", "")
    if not texto.strip():
        print("Texto vazio ou inválido")
        return "Transcrição vazia ou inválida.", ""

    try:
        client, model_id = config_ai()
        if client is None or model_id is None:
            print("Falha na configuração da API")
            return "Erro ao configurar a API do Gemini.", ""

        resumo = agente_resumidor(texto, model_id, client)
        print(f"Resumo gerado: {resumo[:100]}...")
        palavra_chave = extrair_termo_principal(resumo, idioma)
        print(f"Palavra-chave extraída: {palavra_chave}")
        return dcc.Markdown(resumo), palavra_chave
    except Exception as e:
        print(f"Erro no callback gerar_resumo: {str(e)}")
        return f"Erro ao gerar resumo: {str(e)}", ""

# Callback para analisar sentimentos
@app.callback(
    [Output("sentimentos-texto", "children"), Output("sentimentos-grafico", "figure")],
    [Input("transcricao-store", "data")],
    prevent_initial_call=True
)
def analisar_sentimentos(transcricao_data):
    print("Callback analisar_sentimentos acionado")
    if not transcricao_data or not transcricao_data.get("completed", False):
        print("Transcrição não completa ou inexistente")
        return "Aguardando transcrição...", {}

    texto = transcricao_data.get("texto", "")
    if not texto.strip():
        print("Texto vazio ou inválido")
        return "Transcrição vazia ou inválida.", {}

    try:
        client, model_id = config_ai()
        if client is None or model_id is None:
            print("Falha na configuração da API")
            return "Erro ao configurar a API do Gemini.", {}

        sentimento = agente_analisador_sentimentos(texto, model_id, client)
        print(f"Sentimento gerado: {sentimento}")
        figura = {
            "data": [{"x": ["Sentimento"], "y": [sentimento], "type": "bar"}],
            "layout": {"title": "Análise de Sentimentos"}
        }
        return f"Sentimento: {sentimento}", figura
    except Exception as e:
        print(f"Erro no callback analisar_sentimentos: {str(e)}")
        return f"Erro ao analisar sentimentos: {str(e)}", {}

# Callback para atualizar o campo de tópico com a palavra-chave
@app.callback(
    Output("input-topico-agente", "value"),
    [Input("palavra-chave-store", "data")],
    prevent_initial_call=True
)
def atualizar_topico(palavra_chave):
    print("Callback atualizar_topico acionado")
    if not palavra_chave:
        print("Nenhuma palavra-chave disponível")
        return ""
    print(f"Atualizando tópico com: {palavra_chave}")
    return palavra_chave

# Callback para gerar o post
@app.callback(
    Output("output-agente", "children"),
    [Input("btn-gerar-post", "n_clicks")],
    [State("input-topico-agente", "value")],
    prevent_initial_call=True
)
def gerar_post(n_clicks, topico):
    print("Callback gerar_post acionado")
    if not n_clicks or not topico:
        print("Sem cliques ou tópico vazio")
        return "Por favor, digite um tópico para gerar o post."

    client, model_id = config_ai()
    if client is None or model_id is None:
        print("Falha na configuração da API")
        return "Erro ao configurar a API do Gemini."

    final_post = run_agentes(topico, model_id, client)
    print(f"Post gerado: {final_post[:100]}...")
    return [html.H4(f"Post Gerado para o Tópico: {topico}"), dcc.Markdown(final_post)]

# Executar o servidor Dash
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')