#'//////////////////////////////////////////////////////////////////////////////
#' AUTHOR: Manuel Wick-Eckl
#' CREATED: 14 September 2020
#' MODIFIED: 14 September 2020
#' PURPOSE: Extracts Data from Kaggle
#' Status:
#' Comments:
#'//////////////////////////////////////////////////////////////////////////////
#' GLOBAL OPTIONS:

#'Libraries:
library(tidyverse)
library(readr)
library(readxl)
library(kaggler)
library(fs)
library(future)
library(here)
library(furrr)
library(tensorflow)
library(keras)
library(EBImage)

#'Options
plan(multiprocess)

###Custom Function----
#GGPLOT line graph
stock_plot <- function(data) {
  p <- ggplot() +
    geom_line(aes(x = pull(data[1]), y = as.numeric(pull(data[2])))) +
    theme(
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.background = element_blank(),
      plot.margin = margin(0,0,0,0)
    )
}
#Function reads Data and generates line Plots for each selected collum
generate_data <- function(ind, file_path, anzahl_Tage, dir, spalten = c("Open", "High", "Low", "Close", "Volume")) {
  read_quiet <- quietly(read_csv)
  stock <- read_quiet(file_path[ind])
  stock <- stock$result
  stock_name <- str_extract(str_extract(string = file_path[ind], "[^/]*$"), "[[:graph:]]*(?=\\.csv)")

  if (nrow(stock) != 0) {
    start_Tag <- sample(seq(1:nrow(stock) - (anzahl_Tage + 30)), 1)
    end_Tag <- start_Tag + (anzahl_Tage - 1)

    if (nrow(stock) > anzahl_Tage + 30) {
      stock_temp <- stock[seq(start_Tag, end_Tag), ]

      if ( (all(is.na(stock_temp[, spalten]) == FALSE) == TRUE) &
           (all(is.null(stock_temp[, spalten]) == FALSE) == TRUE)
           ){
        stock_temp <- map(spalten, ~ select(stock_temp, Date, .x))

        walk(spalten, ~ fs::dir_create(fs::path(dir, .x)))
        filename <- map(spalten, ~ fs::file_temp(tmp_dir = fs::path(dir, .x), ext = ".jpg"))

        plot <- map(stock_temp, stock_plot)
        walk2(.x = plot, .y = filename, ~ ggsave(
          plot = .x,
          filename = .y,
          height = 1,
          width = 1,
          units = "in",
          dpi = 100,
          device = "jpg"
        ))
        filename <- str_remove(filename, pattern = here())

        dat <- list(
          name = stock_name,
          datum_beginn = stock$Date[start_Tag],
          datum_ende = stock$Date[start_Tag + (anzahl_Tage - 1)],
          anzahl_Tage = anzahl_Tage,
          plots = as.list(filename),
          data_ende = stock[end_Tag, ],
          predict_data = stock[seq(end_Tag + 1, end_Tag + 30), ]
        )

        return(dat)
      }
    } else {
      return(NULL)
    }
  }
}
load_data <- function(dat_gen, ind, spalten = c("Open", "High", "Low", "Close", "Volume"))  {

  path <- pluck(dat_gen$plots[ind]) %>%
    unlist() %>%
    here()

  if (length(path) == length(spalten)){
    channels <- length(path)

    img <- readImage(path ,all = TRUE, names = spalten)
    img <- channel(img, "grey")
    img <- tensorflow::array_reshape(img, c(1,100,100,channels))
    return(img)
  } else {return(NULL)}
}

###Data Preparation----

#kaggle datasets download -d aceofit/stockmarketdatafrom1996to2020
# Dataset downloaded and extractet from kaggle
#104k files
#/Data/Data/Stock/Stock.csv
#Version 1

kaggler::kgl_auth()
data_set_information <- kaggler::kgl_datasets_view(owner_dataset = "aceofit/stockmarketdatafrom1996to2020")
#Check if Kaggledata has changed
data_set_information$currentVersionNumber == 1

data_path <-"01_data/848508_1447516_bundle_archive/Data/Data/"
ticker <- read_excel(here("01_data/848508_1447516_bundle_archive/Tickers.xlsx"))

####Data Generation Part1-Image generation----

time <- paste0(format(lubridate::ymd_hms(Sys.time()), format = "%y%m%d%_%H%M%S"),"_")
dir <- temp_path <- fs::file_temp(pattern = time, tmp_dir =here::here("01_data"))
fs::dir_create(dir)

files <- fs::dir_ls(path = here(data_path), glob = "*.csv", recurse = TRUE)

parts <- seq(1, 20)
for (part in parts) {
  print(part)
  dat_stock_plots_orig <- future_map(sample(x = seq(1, length(files)), size = 10000, replace = TRUE) ,
                                     ~ generate_data(ind = .x, file_path = files, anzahl_Tage = 200, dir = dir) )
  saveRDS(file = path(dir, glue::glue("prep_data_genImage_Part1_{part}.rds")), object = dat_stock_plots_orig)
}

for (part in parts) {
  print(part)
  dat_stock_plots <- read_rds(path(dir, glue::glue("prep_data_genImage_Part1_{part}.rds"))) %>%
    compact() %>%
    purrr::transpose() %>%
    as_tibble()

  #### Data Generation Part2-Data generation----

  img_array <- future_map(seq(1, nrow(dat_stock_plots)), ~ load_data(dat_gen = dat_stock_plots, ind = .x))
  img_array <- compact(img_array)
  img_array <- abind::abind(img_array, along = 1)

  saveRDS(file = path(dir, glue::glue("prep_data_genImage_Part2_{part}.rds")), object = img_array)

  #### Data Generation Part3-Test-Train-Data---
  # Classification labels 0=Stock is lower or equal than yesterday, 1=Stock is higher than yesterday

  temp_train_y <- dat_stock_plots %>%
    hoist(data_ende, close = "Close") %>%
    hoist(predict_data, close_pred = list("Close", 1), .remove = FALSE) %>%
    mutate(
      "close" = as.numeric(close),
      "close_pred" = as.numeric(close_pred)
    ) %>%
    mutate(label = if_else(close >= close_pred, 0, 1))

  ind <- which(is.na(temp_train_y$label))
  img_array <- img_array[-ind, , , ]
  temp_train_y <- temp_train_y[-ind, ]

  # Train, Test, Validation Split
  # 0,70, 0,20, 0,10

  # Test Data
  sample <- caTools::sample.split(temp_train_y$label, SplitRatio = .70)
  ind_test <- which(sample == FALSE)
  test_x <- img_array[ind_test, , , ]
  test_y <- temp_train_y$label[ind_test]

  temp_train_y <- temp_train_y[-ind_test, ]
  img_array <- img_array[-ind_test, , , ]

  # Train and Val Data
  sample <- caTools::sample.split(temp_train_y$label, SplitRatio = .90)
  ind_train <- which(sample == TRUE)
  train_x <- img_array[ind_train, , , ]
  train_y <- temp_train_y$label[ind_train]

  # Validation
  ind_val <- which(sample == FALSE)
  val_x <- img_array[ind_val, , , ]
  val_y <- temp_train_y$label[ind_val]

  #### Data Generation Part4-pack Data together---

  stock_classification_data <- list(
    train_x = train_x,
    train_y = train_y,
    test_x = test_x,
    test_y = test_y,
    val_x = val_x,
    val_y = val_y
  )
  saveRDS(file = path(dir, glue::glue("stock_classification_data_{part}.rds")), object = stock_classification_data)
  gc(verbose = TRUE)
}
