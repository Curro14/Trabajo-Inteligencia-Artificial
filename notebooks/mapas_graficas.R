# ====================================================================
# ANÁLISIS ESPACIAL: SEVILLA COMPLETO (Renta vs Airbnb)
# ====================================================================

library(sf)
library(ggplot2)
library(dplyr)
library(readr)
library(leaflet)
library(htmlwidgets)
library(crosstalk) # Fundamental para añadir los filtros interactivos al HTML
library(htmltools) # Para maquetar la vista del mapa y los filtros

# 1. Carga de datos
ruta_airbnb <- "../data/listings_limpio.csv"
ruta_renta <- "../data/renta_sevilla_capital_limpio.csv"
ruta_poligonos <- "../data/mapa_sevilla.geojson"

airbnb_df <- read_csv(ruta_airbnb, show_col_types = FALSE)
renta_df <- read_csv(ruta_renta, show_col_types = FALSE)
mapa_base <- st_read(ruta_poligonos, quiet = TRUE)

# ====================================================================
# 2. Preprocesamiento
# ====================================================================

mapa_base$CDIS_num <- as.numeric(as.character(mapa_base$CDIS))
renta_df$Distritos_num <- as.numeric(sub(".*distrito (\\d+).*", "\\1", 
                                         as.character(renta_df$Distritos), 
                                         ignore.case = TRUE))

renta_df$Total_num <- as.numeric(gsub("\\.", "", as.character(renta_df$Total)) %>% gsub(",", ".", .))

renta_resumen <- renta_df %>%
  group_by(Distritos_num) %>%
  summarise(Renta_Media = mean(Total_num, na.rm = TRUE))

# Limpio el precio de Airbnb por si viene con simbolos de dolar o comas para poder usarlo en el slider
airbnb_df$price_num <- as.numeric(gsub("[^0-9.]", "", as.character(airbnb_df$price)))

mapa_final_sf <- mapa_base %>%
  left_join(renta_resumen, by = c("CDIS_num" = "Distritos_num")) %>%
  st_transform(4326)

airbnb_final_sf <- airbnb_df %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# ====================================================================
# 3. MAPA ESTÁTICO (PNG) - ENCUADRE MUNICIPAL COMPLETO
# ====================================================================

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
# 4. MAPA DINÁMICO (HTML) CON FILTROS CROSSTALK
# ====================================================================

renta_para_mapa <- unname(mapa_final_sf$Renta_Media)
pal <- colorNumeric("YlGnBu", domain = renta_para_mapa, na.color = "#F2F4F4")

# Creo el objeto SharedData. Esto es lo que permite que los filtros hablen con el mapa en el navegador del cliente.
datos_compartidos <- SharedData$new(airbnb_final_sf)

# Monto el mapa base de Leaflet pero pasandole el objeto compartido en vez del dataframe normal
mapa_leaflet <- leaflet(datos_compartidos, width = "100%", height = 600) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  fitBounds(lng1 = limites["xmin"], lat1 = limites["ymin"],
            lng2 = limites["xmax"], lat2 = limites["ymax"]) %>%
  addPolygons(data = mapa_final_sf,
              fillColor = ~pal(Renta_Media),
              weight = 1, color = "white", fillOpacity = 0.6,
              popup = ~paste("Distrito:", CDIS_num, "<br>Renta:", round(Renta_Media, 2), "€")) %>%
  # Es importante que el circle marker tire de los datos_compartidos para que se oculten al filtrar
  addCircleMarkers(radius = 3, color = "#E74C3C", stroke = FALSE, fillOpacity = 0.6,
                   popup = ~paste("Precio:", price_num, "€<br>Tipo:", room_type, "<br>Capacidad:", accommodates)) %>%
  addLegend(pal = pal, 
            values = renta_para_mapa,
            opacity = 0.7, 
            title = "Renta (€)", 
            position = "bottomright")

# Defino los controles de la interfaz de usuario
filtros_ui <- tagList(
  tags$h3("Filtros del mercado"),
  filter_slider("precio", "Precio por noche (€)", datos_compartidos, ~price_num, width = "100%"),
  filter_select("tipo", "Tipo de alojamiento", datos_compartidos, ~room_type, multiple = TRUE),
  filter_slider("capacidad", "Capacidad (Huéspedes)", datos_compartidos, ~accommodates, width = "100%", step = 1)
)

# Empaqueto los filtros y el mapa en un layout de columnas usando bscols
mapa_dinamico_final <- bscols(
  widths = c(3, 9), # 3 columnas para filtros, 9 para el mapa (de un grid de 12)
  filtros_ui,
  mapa_leaflet
)

# Guardo el resultado. Uso save_html porque el objeto final es una estructura HTML de