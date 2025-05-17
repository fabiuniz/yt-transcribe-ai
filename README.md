# 🎤 Extração de Áudio do YouTube e Processamento via IA

## 📌 Descrição
Este projeto, desenvolvido no ambiente Google Colab, extrai áudio de vídeos do YouTube, transcreve-o usando o modelo Whisper da OpenAI, analisa sentimentos, gera resumos automáticos, extrai palavras-chave e sugere temas para criar posts otimizados para Instagram utilizando agentes de IA baseados na API Gemini. A palavra-chave extraída do resumo é usada como base para o tópico do post, integrando transcrição, análise e criação de conteúdo. A interface web interativa, construída com Dash, facilita a interação com essas funcionalidades. O Google Colab é o ambiente recomendado para a primeira experiência devido à sua facilidade de configuração e suporte a bibliotecas.

## 🚀 Objetivos
- ✅ Extrair áudio de vídeos do YouTube.
- ✅ Transcrever áudio em texto com suporte a inglês, português e espanhol.
- ✅ Gerar legendas sincronizadas no formato SRT.
- ✅ Analisar sentimentos no texto transcrito (positivo, neutro, negativo).
- ✅ Produzir resumos automáticos de até 100 palavras.
- ✅ Extrair palavras-chave relevantes do resumo para sugerir temas de posts.
- ✅ Criar posts engajadores para Instagram com base nos temas extraídos.

## 🔧 Tecnologias Utilizadas
- **Linguagem**: Python
- **Extração de Áudio**: `yt-dlp` (com `pydub` como dependência indireta)
- **Transcrição**: `openai-whisper`
- **Processamento de Texto**: `nltk`, `re`, `collections`
- **Interface Web**: `dash`, `dash-bootstrap-components`
- **IA Generativa**: `google-genai`, `google-adk`, `google_search`
- **Framework de ML**: `torch`
- **Outras**: `requests`, `datetime`
- **Ambiente de Desenvolvimento**: Google Colab (recomendado para a primeira experiência, com execução local opcional)

## ⚙️ Como Rodar o Projeto
O projeto foi desenvolvido no Google Colab, que é o ambiente ideal para a primeira experiência devido à sua infraestrutura pré-configurada e suporte a bibliotecas como `yt-dlp`, `whisper`, e `dash`. A execução local também é suportada.

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/fabiuniz/video_transcription.git
   ```
2. **Instale as dependências**:
   ```bash
   pip install dash dash-bootstrap-components openai-whisper yt-dlp torch nltk requests google-genai google-adk
   ```
3. **Configure a chave da API do Google**:
   - Obtenha uma chave em [Google Cloud Console](https://console.cloud.google.com/).
   - No Google Colab (recomendado):
     ```python
     from google.colab import userdata
     userdata.set('GOOGLE_API_KEY', 'sua-chave-aqui')
     ```
   - Para execução local, defina a variável de ambiente:
     ```bash
     export GOOGLE_API_KEY='sua-chave-aqui'
     ```
4. **Execute no Google Colab (recomendado para iniciantes)**:
   - Crie um novo notebook no [Google Colab](https://colab.research.google.com/).
   - Copie o código de `video_transcription_dash_ai.py` para uma célula.
   - Instale `ngrok` para expor o servidor Dash:
     ```bash
     !pip install pyngrok
     ```
   - Configure e inicie o servidor:
     ```python
     from pyngrok import ngrok
     ngrok.set_auth_token('seu-token-ngrok')
     public_url = ngrok.connect(8050)
     print(f"Acesse em: {public_url}")
     ```
   - Execute o código e acesse a interface via o URL fornecido.
5. **Execute localmente (alternativa)**:
   ```bash
   python video_transcription_dash_ai.py
   ```
   - Acesse a interface em `http://localhost:8050`.

## 📢 Funcionalidades
- **Extração de Áudio**: Baixa áudio de vídeos do YouTube em formato WAV.
- **Transcrição**: Converte áudio em texto com suporte a inglês, português e espanhol.
- **Legendas Sincronizadas**: Exporta transcrições no formato SRT.
- **Análise de Sentimentos**: Classifica o texto transcrito como positivo, neutro ou negativo.
- **Resumo Automático**: Gera resumos concisos (até 100 palavras) do conteúdo transcrito.
- **Extração de Palavras-Chave**: Identifica termos principais do resumo para sugerir temas de posts.
- **Geração de Posts**: Cria posts para Instagram com base no tema extraído, usando agentes de busca, planejamento, redação e revisão.

## 🛠 Passos Implementados
1. **Instalação de Dependências**: Configura bibliotecas como `yt-dlp`, `openai-whisper`, e `dash`.
2. **Extração de Áudio**: Usa `yt-dlp` para baixar e converter áudio de vídeos do YouTube.
3. **Transcrição**: Aplica o modelo Whisper para converter áudio em texto.
4. **Processamento de Texto**: Analisa sentimentos, gera resumos e extrai palavras-chave do resumo para sugerir temas de posts.
5. **Geração de Posts**: Utiliza uma cadeia de agentes (busca, planejamento, redação, revisão) para criar posts otimizados com base no tema extraído.
6. **Interface Web**: Exibe resultados em uma interface Dash com abas para transcrição, resumo, sentimentos e posts.

## 🔥 Exemplo de Código
```python
import yt_dlp
import whisper

# Baixar áudio do YouTube
url = "https://www.youtube.com/watch?v=SEU_VIDEO"
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'audio.wav',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# Transcrever áudio
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])
```

## ✨ Exemplos de Uso
- **Transcrição de Entrevistas**: Converta entrevistas em texto para análise ou estudo.
- **Legendas para Vídeos**: Crie legendas sincronizadas para conteúdos educativos.
- **Resumos de Palestras**: Gere resumos de palestras para facilitar o aprendizado.
- **Análise de Podcasts**: Identifique sentimentos em podcasts para insights de mercado.
- **Posts para Instagram**: Crie posts baseados em temas extraídos de vídeos.
- **Otimização de Conteúdo**: Use palavras-chave para melhorar a relevância de posts.

## 🎯 Melhorias Futuras
- Adicionar suporte a mais idiomas na transcrição (ex.: francês, alemão).
- Integrar APIs de tradução automática (ex.: Google Translate) para transcrições multilíngues.
- Implementar um chatbot para responder perguntas sobre o conteúdo transcrito.
- Otimizar o processamento de áudios longos (ex.: divisão em segmentos).
- Criar um painel de controle avançado para visualizar legendas e status em tempo real.
- Analisar emoções diretamente na fala, além do texto transcrito.
- Formalizar usos práticos para educação, acessibilidade e negócios.
- Exportar resumos em formatos adicionais (ex.: PDF, DOCX).
- Adicionar suporte a vídeos ao vivo do YouTube.
- Criar documentação detalhada com tutoriais e exemplos avançados.

## 📌 Checklist de Melhorias
### ✅ Pontos Fortes
- [x] Uso eficiente de IA para transcrição, análise e geração de conteúdo.
- [x] Interface intuitiva com Dash para interação amigável.
- [x] Código modular e escalável para expansões futuras.
- [x] Automação completa de extração, transcrição e criação de posts.

### 🚀 Pontos a Melhorar
- [ ] Suporte a vídeos ao vivo.
- [ ] Otimização para áudios longos.
- [ ] Integração com APIs de tradução.
- [ ] Documentação detalhada com tutoriais.
- [ ] Exportação de resumos em formatos variados.
- [ ] Análise de emoções na fala.
- [ ] Chatbot interativo.

## 💡 Contribuições
Fique à vontade para sugerir melhorias! Abra uma **issue** ou envie um **pull request** para colaborar.