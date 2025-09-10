# -------------------------------------------------------------------------
# Definir funciones locales -----------
install_if_needed <- function(pk) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}


# -------------------------------------------------------------------------
# Instalar GGIR ---------------------------------------------------
if (!requireNamespace("GGIR", quietly = TRUE)) {
  install.packages("GGIR", repos = "https://cloud.r-project.org")
}

# Cargar los paquetes
library(GGIR)


# -------------------------------------------------------------------------
# Procesar archivo --------------------------------------------------------

# Leer argumentos desde la línea de comandos
args = commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("Please provide the path to a valid accelerometer raw-data file.")
}

input_file = args[1]
# Verificar que el archivo existe
if (!file.exists(input_file)) {
  stop(paste("File not found:", input_file))
}

# Procesar archivo con GGIR
# Esta creacion de carpeta es innecesaria
#suppressWarnings(dir.create("output_GGIR"))
GGIR(mode = 1,
     datadir = input_file, outputdir = "./media/outputs", studyname = basename(input_file),               
     # metrics
     windowsizes = c(1, 900, 3600),
     do.enmo = T, 
     do.anglex = T, do.angley = T, do.anglez = T,
     do.mad = T,
     do.neishabouricounts = T)

# Cargar datos necesarios para la app
outputdir = file.path("media", "outputs", paste0("output_", basename(input_file)), "meta", "basic")
outputfile = dir(outputdir, full.names = T)[1]
cat("Archivo RData generado:", outputfile, "\n")
load(outputfile)
metashort <- M$metashort

# Forzar evaluación completa
metashort <- as.data.frame(metashort)
metashort[] <- lapply(metashort, force)

# Opcional: eliminar tzone si la columna existe
if ("timestamp" %in% colnames(metashort)) {
  attr(metashort$timestamp, "tzone") <- NULL
}
save(metashort, file = file.path("media", "outputs", paste0("metashort_", basename(input_file), ".RData")))

# outputfile contiene (entre otras cosas):
# M: lista con:
#   - metashort = base de datos con métricas (resolución: 1 segundo)
#   - metalong = base de datos con indicador de validez de datos (resolución: 15 minutos)

# I: lista con:
#   - monn = nombre de la marca de acelerómetro
#   - sf = frecuencia de muestreo