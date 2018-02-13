library(RMySQL)
library(psych)

con <- dbConnect(MySQL(),
                 user = 'wspider',
                 password = 'wspider00!q',
                 host = '133.186.143.65',
                 dbname='wspider')
# dbWriteTable(conn = con, name = 'Test', value = as.data.frame(Thurstone))

data <- dbReadTable(conn = con, name = "MWS_COLT_BRAND")

data
