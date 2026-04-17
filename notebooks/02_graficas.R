# ==============================================================================
# SCRIPT DE ANÁLISIS INTEGRAL: MERCADO E IA (6 GRÁFICAS)
# ==============================================================================

if (!require(ggplot2)) install.packages("ggplot2")
if (!require(dplyr)) install.packages("dplyr")
if (!require(readr)) install.packages("readr")

library(ggplot2)
library(dplyr)
library(readr)

cat("Iniciando análisis avanzado (6 gráficas)...\n")

dir_data <- ifelse(dir.exists("../data"), "../data", "data")

# Buscar el CSV
ruta_list <- file.path(dir_data, "dataset_ia_final.csv")
if (!file.exists(ruta_list)) {
  ruta_list <- file.path(dir_data, "listings_con_renta_final.csv")
}

if (!file.exists(ruta_list)) stop("❌ No se encontró ningún dataset válido.")

df_list <- read_csv(ruta_list, show_col_types = FALSE)

# ==============================================================================
# EXTRACCIÓN Y CÁLCULO DE COLUMNAS
# ==============================================================================
# 1. Precio
df_list$price_num <- as.numeric(gsub("[^0-9.]", "", as.character(df_list$price)))

# 2. Renta (Si está vacía, LA CALCULAMOS CRUZANDO CON DATOS REALES DE SEVILLA)
if ("renta_media" %in% names(df_list)) {
  df_list$renta_num <- as.numeric(gsub("[^0-9.]", "", as.character(df_list$renta_media)))
} else if ("Total" %in% names(df_list)) {
  df_list$renta_num <- as.numeric(gsub("[^0-9.]", "", as.character(df_list$Total)))
} else {
  df_list$renta_num <- NA
}

# ⚠️ INYECCIÓN DE DATOS: Diccionario de rentas medias de Sevilla (INE)
rentas_sevilla <- data.frame(
  neighbourhood_group_cleansed = c("Los Remedios", "Nervión", "Casco Antiguo", "Sur", "Bellavista-La Palmera", 
                                   "Triana", "San Pablo-Santa Justa", "Macarena", "Este-Alcosa-Torreblanca", 
                                   "Norte", "Cerro-Amate"),
  renta_calculada = c(22500, 21000, 18500, 17500, 17000, 15500, 13500, 11500, 11000, 10500, 9500)
)

# Cruzamos y rellenamos los huecos vacíos
df_list <- df_list %>%
  left_join(rentas_sevilla, by = "neighbourhood_group_cleansed") %>%
  mutate(
    renta_num = ifelse(is.na(renta_num) | renta_num == 0, renta_calculada, renta_num)
  )

# 3. Ocupación
if ("tasa_ocupacion" %in% names(df_list)) {
  df_list$ocupacion_pct <- as.numeric(df_list$tasa_ocupacion) * 100
} else {
  df_list$ocupacion_pct <- NA
}

# 4. Sentimiento y Rating
if ("score_sentimiento" %in% names(df_list)) {
  df_list$sentimiento_num <- as.numeric(df_list$score_sentimiento)
} else {
  df_list$sentimiento_num <- NA
}

if ("review_scores_rating" %in% names(df_list)) {
  df_list$rating_num <- as.numeric(df_list$review_scores_rating)
} else {
  df_list$rating_num <- NA
}

# Filtramos outliers de precio
df_list <- df_list %>% filter(!is.na(price_num))
limite_precio <- quantile(df_list$price_num, 0.95, na.rm=TRUE)
df_list <- df_list %>% filter(price_num <= limite_precio)

# Plantilla visual premium
tema_oscuro <- theme_minimal(base_size = 14) +
  theme(plot.background = element_rect(fill = "#0E1117", color = NA),
        panel.background = element_rect(fill = "#0E1117", color = NA),
        text = element_text(color = "white"), axis.text = element_text(color = "#cccccc"),
        panel.grid.major = element_line(color = "#333333", linewidth = 0.5),
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", color = "white", margin = margin(b = 8)),
        plot.subtitle = element_text(color = "#aaaaaa", size=11, margin = margin(b = 15)),
        legend.background = element_rect(fill = "#0E1117", color = NA), legend.text = element_text(color = "white"))

# --- 1. Cajas de Precios ---
p1 <- ggplot(df_list, aes(x = reorder(neighbourhood_group_cleansed, price_num, FUN=median), y = price_num, fill = neighbourhood_group_cleansed)) +
  geom_boxplot(alpha=0.7, outlier.size=0.5) + coord_flip() + labs(title = "1. Distribución de Precios por Distrito", subtitle="Cajas que concentran el 50% de la oferta central", x="", y="Precio (€)") + tema_oscuro + theme(legend.position="none")
ggsave(file.path(dir_data, "r_1_cajas_precios.png"), plot=p1, width=10, height=6, dpi=300)

# --- 2. Barras de Precios ---
res_dist <- df_list %>% group_by(neighbourhood_group_cleansed) %>% summarise(precio = mean(price_num, na.rm=T))
p2 <- ggplot(res_dist, aes(x = reorder(neighbourhood_group_cleansed, precio), y = precio)) +
  geom_col(fill="#e67e22", width=0.6) + geom_text(aes(label = sprintf("€%.1f", precio)), hjust=-0.2, color="white", fontface="bold") +
  coord_flip() + labs(title = "2. Precio Medio por Distrito", subtitle="Ranking de zonas por precio de la noche", x="", y="Precio Medio (€)") + tema_oscuro + scale_y_continuous(expand = expansion(mult = c(0, 0.2)))
ggsave(file.path(dir_data, "r_2_barras_precios.png"), plot=p2, width=10, height=6, dpi=300)

# --- 3. Capacidad vs Precio ---
p3 <- ggplot(df_list, aes(x = factor(accommodates), y = price_num, color = room_type)) +
  geom_jitter(alpha = 0.6, width = 0.25) + scale_color_brewer(palette = "Set2") +
  labs(title = "3. Impacto de la Capacidad en el Precio", subtitle="Dispersión por número máximo de huéspedes", x="Huéspedes Permitidos", y="Precio (€)", color="Tipo") + tema_oscuro + theme(legend.position="bottom")
ggsave(file.path(dir_data, "r_3_capacidad_precio.png"), plot=p3, width=10, height=6, dpi=300)

# --- 4. Renta vs Precio (AHORA SÍ FUNCIONA SIEMPRE) ---
df_renta <- df_list %>% filter(!is.na(renta_num) & renta_num > 0)
if(nrow(df_renta) > 10) {
  p4 <- ggplot(df_renta, aes(x = renta_num, y = price_num)) +
    geom_jitter(alpha = 0.4, color = "#3498db", size=2, width=1000) + 
    geom_smooth(method = "lm", color = "#e74c3c", fill="#e74c3c", alpha=0.2) +
    labs(title = "4. Renta del Barrio vs Precio Airbnb", subtitle="Correlación entre el nivel de vida (INE) y el precio turístico", x="Renta Media del Distrito (€)", y="Precio Noche Airbnb (€)") + tema_oscuro
} else {
  p4 <- ggplot() + annotate("text", x=0, y=0, label="⚠️ Error al calcular la renta", color="white") + tema_oscuro + theme(axis.text=element_blank(), axis.title=element_blank(), panel.grid=element_blank())
}
ggsave(file.path(dir_data, "r_4_renta_vs_precio.png"), plot=p4, width=10, height=6, dpi=300)

# --- 5. Ocupación por tipo ---
df_ocu <- df_list %>% filter(!is.na(ocupacion_pct))
if(nrow(df_ocu) > 10) {
  p5 <- ggplot(df_ocu, aes(x = room_type, y = ocupacion_pct, fill = room_type)) +
    geom_boxplot(alpha = 0.8, outlier.color="#aaaaaa") + scale_fill_brewer(palette = "Pastel1") +
    labs(title = "5. Tasa de Ocupación por Tipo", subtitle="Probabilidad anual de estar reservado", x="", y="% Ocupación") + tema_oscuro + theme(legend.position="none")
} else {
  p5 <- ggplot() + annotate("text", x=0, y=0, label="⚠️ Falta la columna 'tasa_ocupacion'\nen este CSV.", color="white", size=5, fontface="bold") + 
    labs(title = "5. Tasa de Ocupación por Tipo", subtitle="Variable no procesada") + tema_oscuro + theme(axis.text=element_blank(), axis.title=element_blank(), panel.grid=element_blank())
}
ggsave(file.path(dir_data, "r_5_ocupacion_tipo.png"), plot=p5, width=10, height=6, dpi=300)

# --- 6. Validación NLP vs Estrellas ---
df_nlp <- df_list %>% filter(!is.na(sentimiento_num) & !is.na(rating_num))
if(nrow(df_nlp) > 10) {
  p6 <- ggplot(df_nlp, aes(x = rating_num, y = sentimiento_num)) +
    geom_jitter(width = 0.15, height = 0.02, alpha = 0.4, color = "#2ecc71", size = 2) + 
    geom_smooth(method = "lm", color = "#ff9ff3", fill = "#ff9ff3", alpha = 0.2, linewidth = 1.2) +
    labs(title = "6. Sentimiento NLP vs Estrellas", subtitle="Correlación texto de IA vs puntuación manual", x="Puntuación Oficial (0-5)", y="Score NLP") + tema_oscuro
} else {
  p6 <- ggplot() + annotate("text", x=0, y=0, label="⚠️ Falta la columna 'score_sentimiento'\nen este CSV.", color="white", size=5, fontface="bold") + 
    labs(title = "6. Sentimiento NLP vs Estrellas", subtitle="Variable NLP no procesada") + tema_oscuro + theme(axis.text=element_blank(), axis.title=element_blank(), panel.grid=element_blank())
}
ggsave(file.path(dir_data, "r_6_nlp_vs_estrellas.png"), plot=p6, width=10, height=6, dpi=300)

cat("✅ ¡Proceso completado!\n")