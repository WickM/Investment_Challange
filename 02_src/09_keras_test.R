library(tensorflow)
library(keras)

tf_version()
is_keras_available()

#Load Data----
stock_classification_data <- read_rds("03_output/classification_data.rds")

dim (stock_classification_data$train_x)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                input_shape = c(100,100,5)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%

  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%

  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

summary(model)

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>%
  fit(
    x = stock_classification_data$train_x, y = stock_classification_data$train_y,
    validation_data =  list (stock_classification_data$test_x, stock_classification_data$test_y),
    epochs = 10,
    verbose = 2,
  )


model %>% evaluate(stock_classification_data$val_x, stock_classification_data$val_y, verbose = 0)






