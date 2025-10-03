import streamlit as st
import numpy as np
import pandas as pd
import re
import requests
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# FUNÇÕES PRINCIPAIS
# ----------------------

def numero_para_cor(n):
    if n == 0: return 0
    if 1 <= n <= 7: return 1  
    if 8 <= n <= 14: return 2
    return -1

def cor_para_emoji(cor):
    return {0: "⚪", 1: "🔴", 2: "⚫"}.get(cor, "❓")

def parse_input(texto):
    if not texto: return []
    parts = re.split(r'[,\s]+', texto.strip())
    nums = []
    for p in parts:
        if p.isdigit():
            v = int(p)
            if 0 <= v <= 14: nums.append(v)
    return nums

def build_features(numeros, window=8):
    if len(numeros) < window + 1: return pd.DataFrame(), pd.Series(dtype=int)
    cores = [numero_para_cor(n) for n in numeros]
    df = pd.DataFrame({"numero": numeros, "cor": cores})
    for i in range(1, window+1):
        df[f"lag_num_{i}"] = df["numero"].shift(i)
        df[f"lag_cor_{i}"] = df["cor"].shift(i)
    df["mean_5"] = df["numero"].rolling(5).mean().shift(1).fillna(0)
    df["std_5"] = df["numero"].rolling(5).std().shift(1).fillna(0)
    for L in (5,10,20,50):
        df[f"freq_b_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==0)).shift(1).fillna(0)
        df[f"freq_v_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==1)).shift(1).fillna(0)
        df[f"freq_p_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==2)).shift(1).fillna(0)
    for L in (10,20,50):
        df[f"prop_b_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==0)/max(1,len(x))).shift(1).fillna(0)
        df[f"prop_v_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==1)/max(1,len(x))).shift(1).fillna(0)
        df[f"prop_p_{L}"] = df["cor"].rolling(L).apply(lambda x: np.sum(x==2)/max(1,len(x))).shift(1).fillna(0)

    seq_len = []
    cur = None; cnt = 0
    for c in df["cor"]:
        if pd.isna(c): seq_len.append(0); cur=None; cnt=0
        else:
            if cur is None or c != cur: cur = c; cnt = 1
            else: cnt += 1
            seq_len.append(cnt)
    df["seq_len"] = seq_len
    df["long_seq_flag"] = (df["seq_len"] >= 3).astype(int)
    df["last_was_branco"] = (df["cor"].shift(1) == 0).astype(int)
    df["gap_num"] = (df["numero"] - df["numero"].shift(1)).abs().shift(1).fillna(0)

    df = df.dropna().reset_index(drop=True)
    if df.empty: return pd.DataFrame(), pd.Series(dtype=int)
    X = df.drop(columns=["numero","cor"])
    X = X.fillna(0)
    y = df["cor"].astype(int)
    return X, y

def train_models(X, y):
    if X.shape[0] < 10 or len(set(y))<2: return None, None
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                       eval_metric="mlogloss", random_state=42)
    xgb.fit(X, y)
    return rf, xgb

def get_probas(model, X_row):
    if model is None: return np.array([0.0,0.0,0.0])
    try:
        p = model.predict_proba(X_row.values.reshape(1, -1))[0]
        probs = np.zeros(3)
        for idx, cls in enumerate(model.classes_): probs[int(cls)] = float(p[idx])
        return probs
    except Exception: return np.array([0.0,0.0,0.0])

def calc_ev_from_probs(probs, odds):
    return [p*o - (1-p) for p,o in zip(probs, odds)]

def compute_model_weights(numeros, window=8, n_splits=4, min_train=30):
    acc_rf = 0.0; acc_xgb = 0.0; cnt = 0
    N = len(numeros)
    if N < min_train + 5: return 0.5, 0.5
    test_block = max(5, (N - min_train) // n_splits)
    split_starts = list(range(min_train, N - test_block + 1, test_block))[:n_splits]
    for s in split_starts:
        train_nums = numeros[:s]
        test_nums = numeros[s:s+test_block]
        X_tr, y_tr = build_features(train_nums, window=window)
        if X_tr.empty or len(y_tr) < 20: continue
        rf, xgb = train_models(X_tr, y_tr)
        if rf is None and xgb is None: continue
        for i in range(len(test_nums)):
            idx = s + i
            X_all, y_all = build_features(numeros[:idx+1], window=window)
            if X_all.empty: continue
            X_pred = X_all.iloc[[-1]]
            true = numero_para_cor(numeros[idx])
            if rf is not None:
                p_rf = get_probas(rf, X_pred)
                pred_rf = int(np.argmax(p_rf))
                acc_rf += (1 if pred_rf == true else 0)
            if xgb is not None:
                p_xgb = get_probas(xgb, X_pred)
                pred_xgb = int(np.argmax(p_xgb))
                acc_xgb += (1 if pred_xgb == true else 0)
            cnt += 1
    if cnt == 0: return 0.5, 0.5
    w_rf = acc_rf / max(1, acc_rf + acc_xgb)
    w_xgb = acc_xgb / max(1, acc_rf + acc_xgb)
    w_rf = 0.5*0.2 + 0.8*w_rf
    w_xgb = 1.0 - w_rf
    return float(w_rf), float(w_xgb)

# ----------------------
# 🎯 NOVAS FUNÇÕES - API BLAZE CORRIGIDA
# ----------------------

def buscar_dados_blaze():
    """Busca os dados mais recentes da API da Blaze - CORRIGIDA"""
    try:
        url = "https://blaze.com/api/roulette_games/recent"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://blaze.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            dados = response.json()
            
            resultados = []
            for jogo in dados:
                resultados.append({
                    'numero': jogo['roll'],
                    'cor': jogo['color'],  # 0=branco, 1=vermelho, 2=preto
                    'data': jogo['created_at'],
                    'id': jogo['id']
                })
            
            return resultados
        else:
            st.sidebar.error(f"❌ Erro HTTP: {response.status_code}")
            return []
            
    except Exception as e:
        st.sidebar.error(f"❌ Exception: {str(e)}")
        return []

def fazer_previsao_automatica(numeros, odds, ev_threshold=0.05):
    """Faz previsão automática para a próxima jogada"""
    if len(numeros) < 30:
        return None
    
    X_full, y_full = build_features(numeros, window=8)
    rf, xgb = train_models(X_full, y_full)
    
    if X_full.empty or (rf is None and xgb is None):
        return None
    
    w_rf, w_xgb = compute_model_weights(numeros, window=8, n_splits=4, min_train=30)
    X_pred = X_full.iloc[[-1]]
    
    p_rf = get_probas(rf, X_pred) if rf is not None else np.array([0.0, 0.0, 0.0])
    p_xgb = get_probas(xgb, X_pred) if xgb is not None else np.array([0.0, 0.0, 0.0])
    
    p_ens = w_rf * p_rf + w_xgb * p_xgb
    evs = calc_ev_from_probs(p_ens, odds)
    
    best_idx = int(np.argmax(p_ens))
    best_ev = evs[best_idx]
    
    if best_ev >= ev_threshold:
        return {
            'cor': best_idx,
            'cor_nome': {0: "BRANCO", 1: "VERMELHO", 2: "PRETO"}[best_idx],
            'emoji': cor_para_emoji(best_idx),
            'probabilidade': p_ens[best_idx],
            'ev': best_ev,
            'odd': odds[best_idx]
        }
    return None

def processar_aposta_automatica(previsao, valor_aposta, numeros_atuais):
    """Processa uma aposta automática e verifica resultado"""
    if len(numeros_atuais) < 2:
        return None
    
    # O último número é o resultado da aposta anterior
    numero_sorteado = numeros_atuais[-1]
    cor_sorteada = numero_para_cor(numero_sorteado)
    
    # Verificar se acertou
    acertou = (previsao['cor'] == cor_sorteada)
    
    # Calcular lucro
    if acertou:
        lucro = (previsao['odd'] - 1.0) * valor_aposta
        resultado = "GANHOU"
    else:
        lucro = -valor_aposta
        resultado = "PERDEU"
    
    return {
        'acertou': acertou,
        'lucro': lucro,
        'resultado': resultado,
        'numero_sorteado': numero_sorteado,
        'cor_sorteada': cor_para_emoji(cor_sorteada),
        'previsao': previsao
    }

# ----------------------
# 🎯 SISTEMA DE GESTÃO DE BANCA
# ----------------------

class GerenciadorBanca:
    def __init__(self):
        if 'banca' not in st.session_state:
            st.session_state.banca = 1000.0
        if 'apostas_automaticas' not in st.session_state:
            st.session_state.apostas_automaticas = []
        if 'modo_auto' not in st.session_state:
            st.session_state.modo_auto = False
        if 'ultima_aposta_id' not in st.session_state:
            st.session_state.ultima_aposta_id = None
        if 'numeros_auto' not in st.session_state:
            st.session_state.numeros_auto = []
        if 'dados_blaze' not in st.session_state:
            st.session_state.dados_blaze = []
        if 'contador_atualizacoes' not in st.session_state:
            st.session_state.contador_atualizacoes = 0
        if 'auto_running' not in st.session_state:
            st.session_state.auto_running = False
    
    def atualizar_banca(self, valor):
        st.session_state.banca = valor
    
    def adicionar_aposta_automatica(self, previsao, valor_aposta, resultado_aposta):
        """Adiciona uma aposta automática ao histórico"""
        aposta = {
            'data': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'cor_apostada': previsao['cor_nome'],
            'emoji_apostada': previsao['emoji'],
            'valor': valor_aposta,
            'odd': previsao['odd'],
            'probabilidade': previsao['probabilidade'],
            'ev': previsao['ev'],
            'resultado': resultado_aposta['resultado'],
            'numero_sorteado': resultado_aposta['numero_sorteado'],
            'cor_sorteada': resultado_aposta['cor_sorteada'],
            'lucro': resultado_aposta['lucro'],
            'acertou': resultado_aposta['acertou']
        }
        st.session_state.apostas_automaticas.append(aposta)
        st.session_state.banca += resultado_aposta['lucro']
    
    def calcular_estatisticas_automaticas(self):
        if not st.session_state.apostas_automaticas:
            return {
                'total_apostas': 0, 
                'lucro_total': 0.0, 
                'acertos': 0,
                'derrotas': 0,
                'taxa_acerto': 0.0,
                'apostas_hoje': 0,
                'sequencia_atual': 0,
                'maior_sequencia_vitorias': 0,
                'maior_sequencia_derrotas': 0
            }
        
        df = pd.DataFrame(st.session_state.apostas_automaticas)
        total_apostas = len(df)
        lucro_total = df['lucro'].sum()
        acertos = len(df[df['acertou'] == True])
        derrotas = len(df[df['acertou'] == False])
        taxa_acerto = (acertos / total_apostas * 100) if total_apostas > 0 else 0
        
        # Calcular sequências
        sequencia_atual = 0
        maior_sequencia_vitorias = 0
        maior_sequencia_derrotas = 0
        sequencia_vitorias_atual = 0
        sequencia_derrotas_atual = 0
        
        for aposta in st.session_state.apostas_automaticas:
            if aposta['acertou']:
                sequencia_vitorias_atual += 1
                sequencia_derrotas_atual = 0
                maior_sequencia_vitorias = max(maior_sequencia_vitorias, sequencia_vitorias_atual)
            else:
                sequencia_derrotas_atual += 1
                sequencia_vitorias_atual = 0
                maior_sequencia_derrotas = max(maior_sequencia_derrotas, sequencia_derrotas_atual)
        
        # Sequência atual (últimos resultados)
        if st.session_state.apostas_automaticas:
            ultimo_resultado = st.session_state.apostas_automaticas[-1]['acertou']
            for aposta in reversed(st.session_state.apostas_automaticas):
                if aposta['acertou'] == ultimo_resultado:
                    sequencia_atual += 1
                else:
                    break
        
        # Apostas de hoje
        hoje = datetime.now().strftime('%d/%m/%Y')
        apostas_hoje = len([a for a in st.session_state.apostas_automaticas 
                           if a['data'].startswith(hoje)])
        
        return {
            'total_apostas': total_apostas,
            'lucro_total': lucro_total,
            'acertos': acertos,
            'derrotas': derrotas,
            'taxa_acerto': taxa_acerto,
            'apostas_hoje': apostas_hoje,
            'sequencia_atual': sequencia_atual,
            'maior_sequencia_vitorias': maior_sequencia_vitorias,
            'maior_sequencia_derrotas': maior_sequencia_derrotas
        }

# ----------------------
# STREAMLIT UI - COM ATUALIZAÇÃO AUTOMÁTICA
# ----------------------

st.set_page_config(page_title="Robô Blaze - Apostas Automáticas", layout="wide")
st.title("🤖 ROBÔ BLAZE - APOSTAS AUTOMÁTICAS")

# Inicializar gerenciador de banca
banca = GerenciadorBanca()

# ----------------------
# 🎯 CONTROLE PRINCIPAL DO ROBÔ
# ----------------------

st.sidebar.header("🤖 CONTROLE DO ROBÔ")

# Modo automático
modo_auto = st.sidebar.checkbox("🎯 ATIVAR MODO AUTOMÁTICO", value=st.session_state.modo_auto)
if modo_auto != st.session_state.modo_auto:
    st.session_state.modo_auto = modo_auto
    if modo_auto:
        st.sidebar.success("🤖 ROBÔ ATIVADO - Apostando automaticamente!")
        st.session_state.auto_running = True
    else:
        st.sidebar.info("⏸️ Robô pausado")
        st.session_state.auto_running = False

# Configurações do robô
st.sidebar.header("⚙️ CONFIGURAÇÕES DO ROBÔ")

odds_b = st.sidebar.number_input("Odds Branco", value=14.0, min_value=2.0, step=0.1)
odds_v = st.sidebar.number_input("Odds Vermelho", value=2.0, min_value=1.0, step=0.1)
odds_p = st.sidebar.number_input("Odds Preto", value=2.0, min_value=1.0, step=0.1)
odds = [odds_b, odds_v, odds_p]

ev_threshold = st.sidebar.slider("Limiar EV Mínimo", 0.0, 0.5, 0.05, step=0.01)
valor_aposta = st.sidebar.number_input("Valor da Aposta (R$)", value=10.0, min_value=1.0, step=5.0)

# Intervalo de verificação
intervalo = st.sidebar.selectbox("Intervalo de Verificação (segundos)", [5, 10, 15, 30, 60], index=1)

# ----------------------
# 🎯 SISTEMA DE ATUALIZAÇÃO AUTOMÁTICA - CORRIGIDO
# ----------------------

def atualizar_dados_automaticamente():
    """Função principal de atualização automática - CORRIGIDO"""
    try:
        # Buscar dados da Blaze
        dados_blaze = buscar_dados_blaze()
        if not dados_blaze:
            st.sidebar.warning("⚠️ Nenhum dado retornado da API")
            return False
        
        # Extrair números (do mais antigo para o mais recente)
        novos_numeros = [jogo['roll'] for jogo in reversed(dados_blaze)]
        
        # Atualizar dados
        st.session_state.numeros_auto = novos_numeros
        st.session_state.dados_blaze = dados_blaze
        st.session_state.ultima_atualizacao = datetime.now().strftime("%H:%M:%S")
        st.session_state.contador_atualizacoes += 1
        
        # Fazer previsão se tiver dados suficientes
        if len(novos_numeros) >= 30 and st.session_state.modo_auto:
            previsao = fazer_previsao_automatica(novos_numeros, odds, ev_threshold)
            
            # Verificar se é uma nova aposta (não repetir a mesma)
            ultimo_id = dados_blaze[0]['id'] if dados_blaze else None
            if previsao and st.session_state.ultima_aposta_id != ultimo_id:
                # Processar aposta automática
                resultado = processar_aposta_automatica(previsao, valor_aposta, novos_numeros)
                if resultado:
                    banca.adicionar_aposta_automatica(previsao, valor_aposta, resultado)
                    st.session_state.ultima_aposta_id = ultimo_id
                    st.sidebar.success(f"🎯 Aposta automática: {previsao['cor_nome']}")
                    return True
        
        return True
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro na atualização: {str(e)}")
        return False

# ----------------------
# 🎯 EXECUTAR ATUALIZAÇÃO AUTOMÁTICA
# ----------------------

# Se o modo automático está ativo, executar atualização
if st.session_state.modo_auto:
    # Executar atualização
    with st.spinner(f"🔄 Atualizando automaticamente... ({intervalo}s)"):
        if atualizar_dados_automaticamente():
            st.success("✅ Dados atualizados com sucesso!", icon="✅")
        else:
            st.error("❌ Falha ao atualizar dados", icon="❌")
    
    # Configurar próximo refresh automático
    st.session_state.auto_running = True

# ----------------------
# 🎯 PAINEL DE STATUS EM TEMPO REAL
# ----------------------

st.header("📊 STATUS DO ROBÔ EM TEMPO REAL")

# Colunas de status
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    status = "🟢 ATIVO" if st.session_state.modo_auto else "🔴 PARADO"
    st.metric("STATUS ROBÔ", status)

with col2:
    if 'ultima_atualizacao' in st.session_state:
        st.metric("🕒 ÚLTIMA ATUALIZAÇÃO", st.session_state.ultima_atualizacao)
    else:
        st.metric("🕒 ÚLTIMA ATUALIZAÇÃO", "--:--:--")

with col3:
    st.metric("💰 BANCA ATUAL", f"R$ {st.session_state.banca:.2f}")

with col4:
    stats = banca.calcular_estatisticas_automaticas()
    st.metric("📊 APOSTAS HOJE", stats['apostas_hoje'])

with col5:
    st.metric("🎯 TAXA ACERTO", f"{stats['taxa_acerto']:.1f}%")

# Contador de atualizações
if 'contador_atualizacoes' in st.session_state:
    st.sidebar.info(f"🔄 Atualizações: {st.session_state.contador_atualizacoes}")

# ----------------------
# 🎯 BOTÕES DE CONTROLE
# ----------------------

col_atualizar, col_estatisticas, col_limpar = st.columns(3)

with col_atualizar:
    if st.button("🔄 ATUALIZAR DADOS AGORA", type="primary"):
        with st.spinner("Buscando dados da Blaze..."):
            if atualizar_dados_automaticamente():
                st.success("✅ Dados atualizados com sucesso!")
            else:
                st.error("❌ Falha ao atualizar dados")

with col_estatisticas:
    if st.button("📊 ATUALIZAR ESTATÍSTICAS"):
        st.rerun()

with col_limpar:
    if st.button("🗑️ LIMPAR HISTÓRICO"):
        st.session_state.apostas_automaticas = []
        st.session_state.ultima_aposta_id = None
        st.session_state.contador_atualizacoes = 0
        st.success("✅ Histórico limpo!")
        st.rerun()

# ----------------------
# 🎯 DADOS ATUAIS DA BLAZE - CORRIGIDO
# ----------------------

if 'dados_blaze' in st.session_state and st.session_state.dados_blaze:
    st.header("🎰 ÚLTIMOS RESULTADOS BLAZE")
    
    # Mostrar últimos 10 resultados
    ultimos_resultados = st.session_state.dados_blaze[:10]
    
    for jogo in ultimos_resultados:
        cor_emoji = cor_para_emoji(jogo['color'])
        hora = jogo['data'][11:19] if 'data' in jogo else "N/A"
        st.write(f"{cor_emoji} **Número {jogo['numero']}** - {hora}")

# ----------------------
# 🎯 PRÓXIMA PREVISÃO
# ----------------------

st.header("🎯 PRÓXIMA PREVISÃO")

if 'numeros_auto' in st.session_state and len(st.session_state.numeros_auto) >= 30:
    previsao = fazer_previsao_automatica(st.session_state.numeros_auto, odds, ev_threshold)
    
    if previsao:
        cor_bg = {"VERMELHO": "#ff4d4d", "PRETO": "#262626", "BRANCO": "#f2f2f2"}[previsao['cor_nome']]
        text_color = "white" if previsao['cor_nome'] in ["VERMELHO", "PRETO"] else "black"
        
        st.markdown(f"""
        <div style="background-color: {cor_bg}; padding: 20px; border-radius: 10px; text-align: center; color: {text_color};">
            <h3>🎯 PRÓXIMA APOSTA RECOMENDADA</h3>
            <h1 style="font-size: 2.5em; margin: 10px 0;">{previsao['cor_nome']} {previsao['emoji']}</h1>
            <div style="font-size: 1.2em;">
                Probabilidade: <b>{previsao['probabilidade']*100:.1f}%</b> | 
                EV: <b>{previsao['ev']:+.3f}</b> | 
                Odd: <b>{previsao['odd']}x</b>
            </div>
            <div style="font-size: 1.1em; margin-top: 10px;">
                Valor Aposta: <b>R$ {valor_aposta:.2f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.modo_auto:
            st.success("🤖 Robô apostará automaticamente quando o EV for favorável!")
    else:
        st.warning("⚠️ Nenhuma entrada segura no momento - EV abaixo do limiar")
else:
    dados_coletados = len(st.session_state.numeros_auto) if 'numeros_auto' in st.session_state else 0
    st.info(f"📊 Coletando dados... {dados_coletados}/30 rodadas para previsões.")

# ----------------------
# 🎯 ACERTOS E DERROTAS DETALHADOS
# ----------------------

st.header("📈 ACERTOS E DERROTAS DETALHADOS")

stats = banca.calcular_estatisticas_automaticas()

# Métricas principais de performance
col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

with col_perf1:
    st.metric("✅ ACERTOS", stats['acertos'], delta=f"{stats['taxa_acerto']:.1f}%")

with col_perf2:
    st.metric("❌ DERROTAS", stats['derrotas'], delta=f"{(100 - stats['taxa_acerto']):.1f}%", delta_color="inverse")

with col_perf3:
    st.metric("💰 LUCRO TOTAL", f"R$ {stats['lucro_total']:.2f}")

with col_perf4:
    st.metric("📊 TOTAL APOSTAS", stats['total_apostas'])

# Sequências e estatísticas avançadas
st.subheader("📊 ESTATÍSTICAS DE SEQUÊNCIAS")

col_seq1, col_seq2, col_seq3 = st.columns(3)

with col_seq1:
    # Sequência atual
    if st.session_state.apostas_automaticas:
        ultimo_resultado = st.session_state.apostas_automaticas[-1]['acertou']
        emoji_seq = "✅" if ultimo_resultado else "❌"
        texto_seq = "Vitórias" if ultimo_resultado else "Derrotas"
        st.metric(f"📈 SEQUÊNCIA ATUAL ({emoji_seq})", f"{stats['sequencia_atual']} {texto_seq}")
    else:
        st.metric("📈 SEQUÊNCIA ATUAL", "0")

with col_seq2:
    st.metric("🏆 MAIOR SEQUÊNCIA DE VITÓRIAS", stats['maior_sequencia_vitorias'])

with col_seq3:
    st.metric("⚡ MAIOR SEQUÊNCIA DE DERROTAS", stats['maior_sequencia_derrotas'])

# Gráfico de pizza - Acertos vs Derrotas
if stats['total_apostas'] > 0:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    labels = ['✅ Acertos', '❌ Derrotas']
    sizes = [stats['acertos'], stats['derrotas']]
    colors = ['#4CAF50', '#F44336']
    explode = (0.1, 0)  # destaca os acertos
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title('Distribuição de Acertos e Derrotas')
    
    st.pyplot(fig1)

# ----------------------
# 🎯 HISTÓRICO DETALHADO DE APOSTAS
# ----------------------

st.header("📋 HISTÓRICO COMPLETO DE APOSTAS")

if st.session_state.apostas_automaticas:
    # Mostrar todas as apostas em ordem reversa (mais recente primeiro)
    for i, aposta in enumerate(reversed(st.session_state.apostas_automaticas)):
        cor_bg = "#d4edda" if aposta['acertou'] else "#f8d7da"
        cor_texto = "#155724" if aposta['acertou'] else "#721c24"
        emoji = "✅" if aposta['acertou'] else "❌"
        borda_cor = "#28a745" if aposta['acertou'] else "#dc3545"
        
        st.markdown(f"""
        <div style="background-color: {cor_bg}; padding: 12px; border-radius: 8px; margin: 8px 0; 
                    border-left: 6px solid {borda_cor}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: {cor_texto}; font-weight: bold; font-size: 1.1em;">
                {emoji} <strong>{aposta['resultado']}</strong> - {aposta['data']}
            </div>
            <div style="color: {cor_texto}; font-size: 0.95em; margin-top: 8px;">
                <strong>Aposta:</strong> {aposta['cor_apostada']} {aposta['emoji_apostada']} | 
                <strong>Valor:</strong> R$ {aposta['valor']:.2f} | 
                <strong>Odd:</strong> {aposta['odd']}x
            </div>
            <div style="color: {cor_texto}; font-size: 0.95em;">
                <strong>Resultado:</strong> Nº {aposta['numero_sorteado']} {aposta['cor_sorteada']} | 
                <strong>Lucro:</strong> <span style="font-weight: bold; color: {'green' if aposta['lucro'] > 0 else 'red'}">
                R$ {aposta['lucro']:.2f}</span> | 
                <strong>EV:</strong> {aposta['ev']:.3f}
            </div>
            <div style="color: {cor_texto}; font-size: 0.9em; margin-top: 4px; font-style: italic;">
                Probabilidade: {aposta['probabilidade']*100:.1f}% | 
                Aposta #{len(st.session_state.apostas_automaticas) - i}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("🤖 Nenhuma aposta automática registrada ainda. Ative o modo automático!")

# ----------------------
# 🎯 GRÁFICO DE EVOLUÇÃO DO LUCRO
# ----------------------

if st.session_state.apostas_automaticas:
    st.header("📊 EVOLUÇÃO DO DESEMPENHO")
    
    df_apostas = pd.DataFrame(st.session_state.apostas_automaticas)
    df_apostas['lucro_acumulado'] = df_apostas['lucro'].cumsum()
    df_apostas['numero_aposta'] = range(1, len(df_apostas) + 1)
    
    # Gráfico de lucro acumulado
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Linha do lucro acumulado
    ax2.plot(df_apostas['numero_aposta'], df_apostas['lucro_acumulado'], 
             color='green', linewidth=3, label='Lucro Acumulado', marker='o')
    
    # Linha de referência zero
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
    
    # Destacar pontos de vitória e derrota
    cores = ['green' if x > 0 else 'red' for x in df_apostas['lucro']]
    ax2.scatter(df_apostas['numero_aposta'], df_apostas['lucro_acumulado'], 
                c=cores, s=50, alpha=0.6)
    
    ax2.set_xlabel("Número da Aposta")
    ax2.set_ylabel("Lucro Acumulado (R$)")
    ax2.set_title("Evolução do Lucro do Robô")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

# ----------------------
# 🎯 SISTEMA DE ATUALIZAÇÃO AUTOMÁTICA COM STREAMLIT
# ----------------------

# Usando o st.rerun() para atualização automática
if st.session_state.modo_auto and st.session_state.auto_running:
    # Mostrar info de atualização
    st.sidebar.info(f"🔄 Próxima atualização em {intervalo} segundos")
    
    # Adicionar um pequeno delay antes do rerun
    time.sleep(1)
    
    # Forçar atualização da página
    st.rerun()

st.markdown("---")
st.caption("🤖 Robô Blaze - Sistema de Apostas Automáticas | Atualização Automática Ativa")
