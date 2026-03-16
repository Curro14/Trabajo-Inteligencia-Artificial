# ====================================================================
# ANÁLISIS ESPACIAL: SEVILLA COMPLETO (Renta vs Airbnb)
# ====================================================================

library(sf)
library(ggplot2)
library(dplyr)
library(readr)
library(leaflet)
library(htmlwidgets)

# 1. Carga de datos
ruta_airbnb <- "../data/listings_limpio.csv"
ruta_renta <- "../data/renta_sevilla_capital_limpio.csv"
ruta_poligonos <- "../data/mapa_sevilla.geojson" # Usamos el mapa municipal completo

airbnb_df <- read_csv(ruta_airbnb, show_col_types = FALSE)
renta_df <- read_csv(ruta_renta, show_col_types = FALSE)
mapa_base <- st_read(ruta_poligonos, quiet = TRUE)

# ====================================================================
# 2. Preprocesamiento (Cero dependencias de Python)
# ====================================================================

# Limpieza de identificadores de distrito
mapa_base$CDIS_num <- as.numeric(as.character(mapa_base$CDIS))
renta_df$Distritos_num <- as.numeric(sub(".*distrito (\\d+).*", "\\1", 
                                         as.character(renta_df$Distritos), 
                                         ignore.case = TRUE))

# Limpieza de valores de renta (manejo de puntos y comas)
renta_df$Total_num <- as.numeric(gsub("\\.", "", as.character(renta_df$Total)) %>% gsub(",", ".", .))

# Agrupación por distrito (Media)
renta_resumen <- renta_df %>%
  group_by(Distritos_num) %>%
  summarise(Renta_Media = mean(Total_num, na.rm = TRUE))

# Join y Proyección a WGS84 (estándar GPS)
mapa_final_sf <- mapa_base %>%
  left_join(renta_resumen, by = c("CDIS_num" = "Distritos_num")) %>%
  st_transform(4326)

airbnb_final_sf <- airbnb_df %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# ====================================================================
# 3. MAPA ESTÁTICO (PNG) - ENCUADRE MUNICIPAL COMPLETO
# ====================================================================

# Calculamos límites basados en el polígono municipal completo
limites <- st_bbox(mapa_final_sf)

mapa_estatico <- ggplot() +
  geom_sf(data = mapa_final_sf, aes(fill = Renta_Media), color = "white", linewidth = 0.1) +
  scale_fill_distiller(palette = "YlGnBu", direction = 1, na.value = "#F2F4F4", name = "Renta (€)") +
  geom_sf(data = airbnb_final_sf, color = "#E74C3C", size = 0.7, alpha = 0.3) +
  coord_sf(xlim = c(limites["xmin"], limites["xmax"]),
           ylim = c(limites["ymin"], limites["ymax"]),
           expand = TRUE) +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(t=20, b=10)),
    legend.position = "bottom",
    legend.key.width = unit(2, "cm")
  ) +
  labs(title = "Sevilla: Nivel de Renta y Alojamientos Turísticos")

ggsave("../data/mapa_estatico_renta_airbnb.png", plot = mapa_estatico, width = 12, height = 10, bg = "white")
# ====================================================================
# 4. MAPA INTERACTIVO (HTML) - SIN AVISOS DE JSON
# ====================================================================

# 1. Limpiamos los nombres de los vectores para evitar los warnings de jsonlite
renta_para_mapa <- unname(mapa_final_sf$Renta_Media)

# 2. Definimos la paleta con el vector limpio
pal <- colorNumeric("YlGnBu", domain = renta_para_mapa, na.color = "#F2F4F4")

mapa_interactivo <- leaflet() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  # Encuadre automático al municipio completo
  fitBounds(lng1 = limites["xmin"], lat1 = limites["ymin"],
            lng2 = limites["xmax"], lat2 = limites["ymax"]) %>%
  addPolygons(data = mapa_final_sf,
              fillColor = ~pal(Renta_Media),
              weight = 1, color = "white", fillOpacity = 0.6,
              popup = ~paste("Distrito:", CDIS_num, "<br>Renta:", round(Renta_Media, 2), "€")) %>%
  addCircleMarkers(data = airbnb_final_sf,
                   radius = 2, color = "#E74C3C", stroke = FALSE, fillOpacity = 0.4) %>%
  addLegend(pal = pal, 
            values = renta_para_mapa, # Usamos el vector sin nombres aquí también
            opacity = 0.7, 
            title = "Renta (€)", 
            position = "bottomright")

# 3. Guardado definitivo
saveWidget(mapa_interactivo, file = "../data/mapa_interactivo_renta.html", selfcontained = TRUE)