import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, t

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Riesgo Financiero", layout="wide")
st.title("Proyecto 1, calculo de Value-At-Risk y de Expected Shortfall.")

def obtener_datos(stocks):
    '''Descarga el precio de cierre de un o varios activos'''
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    '''Calcula los rendimientos de un activo'''
    return df.pct_change().dropna()

def calcular_var_parametrico(returns, alpha, distrib='normal'):
    '''Calcula VaR paramétrico bajo distribución normal o t-Student'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    if distrib == 'normal':
        VaR = norm.ppf(1-alpha, mean, stdev)
    elif distrib == 't':
        df_t = 10
        VaR = t.ppf(1-alpha, df_t, mean, stdev)
    return VaR

def calcular_var_historico(returns, alpha):
    '''Calcula VaR histórico'''
    return returns.quantile(1-alpha)

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    '''Calcula VaR usando simulaciones de Monte Carlo'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, hVaR):
    '''Calcula Conditional Value at Risk (CVaR)'''
    return returns[returns <= hVaR].mean()

# Sidebar para configuración
st.sidebar.header("Configuración")
ticker = st.sidebar.text_input("Ingrese el ticker del activo:", "AAPL")
conf_levels = st.sidebar.multiselect(
    "Niveles de confianza:",
    options=[0.95, 0.975, 0.99],
    default=[0.95, 0.975, 0.99]
)

# Procesamiento principal
if st.sidebar.button("Calcular"):
    with st.spinner("Procesando datos..."):
        # a) Obtener datos y calcular rendimientos
        M7 = [ticker]
        df_precios = obtener_datos(M7)
        df_rendimientos = calcular_rendimientos(df_precios)
        
        # Mostrar datos básicos
        st.subheader("Datos básicos del activo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Precios de cierre**")
            st.dataframe(df_precios.tail())
            
        with col2:
            st.write("**Rendimientos diarios**")
            st.dataframe(df_rendimientos.tail())
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Promedio rendimiento diario", f"{df_rendimientos[ticker].mean():.4%}")
            
        with col2:
            st.metric("Curtosis", f"{kurtosis(df_rendimientos[ticker]):.4f}")
            
        with col3:
            st.metric("Sesgo", f"{skew(df_rendimientos[ticker]):.4f}")
        
        # Cálculos de VaR y CVaR
        st.subheader("Resultados de VaR y CVaR")
        
        resultados_df = pd.DataFrame(columns=['VaR (Normal)', 'VaR (t-Student)', 'VaR (Histórico)', 'VaR (Monte Carlo)', 
                                          'CVaR (Normal)', 'CVaR (t-Student)', 'CVaR (Histórico)', 'CVaR (Monte Carlo)'])
        
        for alpha in conf_levels:
            var_normal = calcular_var_parametrico(df_rendimientos[ticker], alpha, distrib='normal')
            var_t = calcular_var_parametrico(df_rendimientos[ticker], alpha, distrib='t')
            var_historico = calcular_var_historico(df_rendimientos[ticker], alpha)
            var_montecarlo = calcular_var_montecarlo(df_rendimientos[ticker], alpha)
            
            cvar_normal = calcular_cvar(df_rendimientos[ticker], var_normal)
            cvar_t = calcular_cvar(df_rendimientos[ticker], var_t)
            cvar_historico = calcular_cvar(df_rendimientos[ticker], var_historico)
            cvar_montecarlo = calcular_cvar(df_rendimientos[ticker], var_montecarlo)
            
            resultados_df.loc[f'{int(alpha*100)}% Confidence'] = [
                var_normal * 100, var_t * 100, var_historico * 100, var_montecarlo * 100,
                cvar_normal * 100, cvar_t * 100, cvar_historico * 100, cvar_montecarlo * 100
            ]
        
        st.dataframe(resultados_df.style.format("{:.2f}%"))
        
        # Gráfico Histograma
        st.subheader("Histograma de Rendimientos con VaR y CVaR")
        
        alpha = conf_levels[-1]
        var_normal = calcular_var_parametrico(df_rendimientos[ticker], alpha, distrib='normal')
        var_historico = calcular_var_historico(df_rendimientos[ticker], alpha)
        var_montecarlo = calcular_var_montecarlo(df_rendimientos[ticker], alpha)
        cvar_normal = calcular_cvar(df_rendimientos[ticker], var_normal)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax1.hist(df_rendimientos[ticker], bins=50, color='blue', alpha=0.7, label='Returns')
        
        for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
            if bin_left < var_historico:
                patch.set_facecolor('red')
        
        ax1.axvline(x=var_normal, color='skyblue', linestyle='--', label=f'VaR {int(alpha*100)}% (Normal)')
        ax1.axvline(x=var_montecarlo, color='grey', linestyle='--', label=f'VaR {int(alpha*100)}% (Monte Carlo)')
        ax1.axvline(x=var_historico, color='green', linestyle='--', label=f'VaR {int(alpha*100)}% (Histórico)')
        ax1.axvline(x=cvar_normal, color='purple', linestyle='-.', label=f'CVaR {int(alpha*100)}% (Normal)')
        
        ax1.set_title(f'Histograma de Rendimientos con VaR y CVaR ({int(alpha*100)}% Confianza)')
        ax1.set_xlabel('Rendimientos')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        
        st.pyplot(fig1)
        
        # -------------------------------------------------------------------------
        # Inciso D - Rolling VaR y ES
        # -------------------------------------------------------------------------
        st.subheader("Análisis Rolling Window de VaR y ES")
        
        window_size = 252

        rolling_mean = df_rendimientos[ticker].rolling(window=window_size).mean()
        rolling_std = df_rendimientos[ticker].rolling(window=window_size).std()

        VaR_95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)
        VaR_99_rolling = norm.ppf(1-0.99, rolling_mean, rolling_std)

        VaR_95_rolling_hist = df_rendimientos[ticker].rolling(window=window_size).quantile(0.05)
        VaR_99_rolling_hist = df_rendimientos[ticker].rolling(window=window_size).quantile(0.01)

        def calcular_ES(returns, var_rolling):
            return returns[returns <= var_rolling].mean()

        ES_95_rolling = [calcular_ES(df_rendimientos[ticker][i-window_size:i], VaR_95_rolling[i]) for i in range(window_size, len(df_rendimientos))]
        ES_99_rolling = [calcular_ES(df_rendimientos[ticker][i-window_size:i], VaR_99_rolling[i]) for i in range(window_size, len(df_rendimientos))]

        def calcular_ES_hist(returns, var_hist_rolling):
            return returns[returns <= var_hist_rolling].mean()

        ES_95_rolling_hist = [calcular_ES_hist(df_rendimientos[ticker][i-window_size:i], VaR_95_rolling_hist[i]) for i in range(window_size, len(df_rendimientos))]
        ES_99_rolling_hist = [calcular_ES_hist(df_rendimientos[ticker][i-window_size:i], VaR_99_rolling_hist[i]) for i in range(window_size, len(df_rendimientos))]

        rolling_results_df = pd.DataFrame({
            'Date': df_rendimientos.index[window_size:],
            'VaR_95_Rolling': VaR_95_rolling[window_size:],
            'VaR_99_Rolling': VaR_99_rolling[window_size:],
            'VaR_95_Rolling_Hist': VaR_95_rolling_hist[window_size:],
            'VaR_99_Rolling_Hist': VaR_99_rolling_hist[window_size:],
            'ES_95_Rolling': ES_95_rolling,
            'ES_99_Rolling': ES_99_rolling,
            'ES_95_Rolling_Hist': ES_95_rolling_hist,
            'ES_99_Rolling_Hist': ES_99_rolling_hist
        })
        rolling_results_df.set_index('Date', inplace=True)

        st.write("**Resultados Rolling Window (252 días)**")
        st.dataframe(rolling_results_df.tail().style.format("{:.2f}%"))

        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(df_rendimientos.index, df_rendimientos[ticker] * 100, label='Rendimientos Diarios (%)', color='blue', alpha=0.5)
        ax2.plot(rolling_results_df.index, rolling_results_df['VaR_95_Rolling'] * 100, label='VaR 95% Rolling Paramétrico', color='red')
        ax2.plot(rolling_results_df.index, rolling_results_df['VaR_99_Rolling'] * 100, label='VaR 99% Rolling Paramétrico', color='green')
        ax2.plot(rolling_results_df.index, rolling_results_df['VaR_95_Rolling_Hist'] * 100, label='VaR 95% Rolling Histórico', color='orange')
        ax2.plot(rolling_results_df.index, rolling_results_df['VaR_99_Rolling_Hist'] * 100, label='VaR 99% Rolling Histórico', color='purple')
        ax2.plot(rolling_results_df.index, rolling_results_df['ES_95_Rolling'] * 100, label='ES 95% Rolling Paramétrico', color='cyan')
        ax2.plot(rolling_results_df.index, rolling_results_df['ES_99_Rolling'] * 100, label='ES 99% Rolling Paramétrico', color='magenta')
        ax2.plot(rolling_results_df.index, rolling_results_df['ES_95_Rolling_Hist'] * 100, label='ES 95% Rolling Histórico', color='pink')
        ax2.plot(rolling_results_df.index, rolling_results_df['ES_99_Rolling_Hist'] * 100, label='ES 99% Rolling Histórico', color='yellow')
        ax2.set_title('Rendimientos Diarios y VaR/ES Rolling Window (252 días)')
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Valor (%)')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        
        # -------------------------------------------------------------------------
        # Inciso E - Backtesting (Violaciones)
        # -------------------------------------------------------------------------
        st.subheader("Backtesting - Análisis de Violaciones")
        
        def contar_violaciones(returns, risk_measure):
            violations = returns < risk_measure
            num_violations = violations.sum()
            violation_percentage = (num_violations / len(returns)) * 100
            return num_violations, violation_percentage

        returns_for_test = df_rendimientos[ticker][window_size:]

        violation_results = pd.DataFrame(columns=['VaR 95% Paramétrico', 'VaR 99% Paramétrico',
                                                 'VaR 95% Histórico', 'VaR 99% Histórico',
                                                 'ES 95% Paramétrico', 'ES 99% Paramétrico',
                                                 'ES 95% Histórico', 'ES 99% Histórico'],
                                        index=['Número de violaciones', 'Porcentaje de violaciones'])

        var_95_violations, var_95_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_95_Rolling'])
        var_99_violations, var_99_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_99_Rolling'])
        var_95_hist_violations, var_95_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_95_Rolling_Hist'])
        var_99_hist_violations, var_99_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['VaR_99_Rolling_Hist'])

        es_95_violations, es_95_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_95_Rolling'])
        es_99_violations, es_99_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_99_Rolling'])
        es_95_hist_violations, es_95_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_95_Rolling_Hist'])
        es_99_hist_violations, es_99_hist_percent = contar_violaciones(returns_for_test, rolling_results_df['ES_99_Rolling_Hist'])

        violation_results.loc['Número de violaciones'] = [
            var_95_violations, var_99_violations,
            var_95_hist_violations, var_99_hist_violations,
            es_95_violations, es_99_violations,
            es_95_hist_violations, es_99_hist_violations
        ]

        violation_results.loc['Porcentaje de violaciones'] = [
            var_95_percent, var_99_percent,
            var_95_hist_percent, var_99_hist_percent,
            es_95_percent, es_99_percent,
            es_95_hist_percent, es_99_hist_percent
        ]

        st.write("**Resultados de violaciones de VaR y ES:**")
        st.dataframe(violation_results.style.format("{:.2f}%", subset=pd.IndexSlice['Porcentaje de violaciones', :]))
        
        # -------------------------------------------------------------------------
        # Inciso F - VaR con volatilidad móvil
        # -------------------------------------------------------------------------
        st.subheader("VaR con Volatilidad Móvil")
        
        # Definimos los niveles de significancia
        alpha_1 = 0.05  # Para VaR 95%
        alpha_2 = 0.01   # Para VaR 99%

        # Calculamos la desviación estándar móvil de 252 días (volatilidad)
        rolling_volatility = df_rendimientos[ticker].rolling(window=252).std()

        # Calculamos los percentiles de la distribución normal
        q_alpha1 = norm.ppf(alpha_1)
        q_alpha2 = norm.ppf(alpha_2)

        # Calculamos el VaR móvil para ambos niveles de confianza
        VaR_95_vol_movil = q_alpha1 * rolling_volatility
        VaR_99_vol_movil = q_alpha2 * rolling_volatility

        # Graficamos los resultados
        fig3, ax3 = plt.subplots(figsize=(14, 7))

        # Graficar los rendimientos diarios
        ax3.plot(df_rendimientos.index, df_rendimientos[ticker] * 100, 
                label='Rendimientos Diarios (%)', color='blue', alpha=0.3)

        # Graficar el VaR con volatilidad móvil
        ax3.plot(VaR_95_vol_movil.index, VaR_95_vol_movil * 100, 
                label='VaR 95% con Volatilidad Móvil', color='red')
        ax3.plot(VaR_99_vol_movil.index, VaR_99_vol_movil * 100, 
                label='VaR 99% con Volatilidad Móvil', color='darkred')

        # Configuración del gráfico
        ax3.set_title('Rendimientos Diarios y VaR con Volatilidad Móvil (252 días)')
        ax3.set_xlabel('Fecha')
        ax3.set_ylabel('Valor (%)')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        st.pyplot(fig3)

        # Función modificada para contar violaciones que alinea los índices
        def contar_violaciones_alineadas(returns, risk_measure):
            # Convertimos a DataFrame para hacer merge
            df_returns = returns.to_frame(name='returns')
            df_risk = risk_measure.to_frame(name='risk')
            
            # Unimos las series por índice
            merged = pd.merge(df_returns, df_risk, left_index=True, right_index=True)
            
            # Calculamos violaciones
            violations = merged['returns'] < merged['risk']
            num_violations = violations.sum()
            violation_percentage = (num_violations / len(merged)) * 100
            
            return num_violations, violation_percentage

        # Preparamos los datos para el análisis de violaciones
        returns_for_test_vol = df_rendimientos[ticker].loc[VaR_95_vol_movil.dropna().index]

        # Calculamos violaciones con la función modificada
        vol_movil_violations_95, vol_movil_percent_95 = contar_violaciones_alineadas(
            returns_for_test_vol, VaR_95_vol_movil.dropna())
        vol_movil_violations_99, vol_movil_percent_99 = contar_violaciones_alineadas(
            returns_for_test_vol, VaR_99_vol_movil.dropna())

        # Mostramos los resultados de violaciones
        st.write("**Resultados de violaciones para VaR con volatilidad móvil:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("VaR 95% con volatilidad móvil", 
                    f"{vol_movil_violations_95} violaciones", 
                    f"{vol_movil_percent_95:.2f}%")
        with col2:
            st.metric("VaR 99% con volatilidad móvil", 
                    f"{vol_movil_violations_99} violaciones", 
                    f"{vol_movil_percent_99:.2f}%")

        # Evaluación de la calidad de las estimaciones
        st.write("**Evaluación de la calidad de las estimaciones con volatilidad móvil:**")
        st.write(f"Para VaR 95%: esperado ~5% de violaciones, obtenido {vol_movil_percent_95:.2f}%")
        st.write(f"Para VaR 99%: esperado ~1% de violaciones, obtenido {vol_movil_percent_99:.2f}%")
        st.write("**Nota:** Un buen modelo debería tener menos del 2.5% de violaciones en general.")

# Información adicional
st.sidebar.markdown("""
**Instrucciones:**
1. Ingrese el ticker del activo (ej. AAPL para Apple)
2. Seleccione los niveles de confianza deseados
3. Haga clic en 'Calcular' para ver los resultados
""")
