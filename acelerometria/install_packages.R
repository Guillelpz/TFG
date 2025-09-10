required_packages <- c("GGIR", "ggplot2", "data.table", "dplyr", "lifecycle")

for (pkg in required_packages) {
  message(paste0("📦 Installing ", pkg, "..."))
  tryCatch(
    {
      install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    },
    error = function(e) {
      message(paste0("❌ Error installing ", pkg, ": ", e$message))
    }
  )
}

install.packages("devtools", repos = "https://cloud.r-project.org")
library(devtools) 

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org")

BiocManager::install(c("GenomicRanges", "GenomeInfoDb"), ask = FALSE)

install.packages("RCurl", repos = "https://cloud.r-project.org")

devtools::install_github("wadpac/GGIR", dependencies = TRUE, upgrade = "always")
