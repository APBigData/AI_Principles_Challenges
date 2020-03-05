require ( "data.table" )
require ( "magrittr" )
require ( "manipulate" )

setwd ( "C:/Users/David D'Haese/Documents/_TEACHING/AI_PRINCIPLES/COURSE/AI_Principles_Challenges/Air_Quality/dat" )

dat <- fread ( "AirQualityUCI.csv" )

# Removing empty columns
dat$V16 <- NULL
dat$V17 <- NULL

# Slow
# pairs(dat[,.(`CO(GT)`, `PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`,
#              `PT08.S2(NMHC)`, AH, RH, `T`, `PT08.S5(O3)`,
#              `PT08.S4(NO2)`, `NO2(GT)`, `PT08.S3(NOx)`,
#              `NOx(GT)`)], pch=".", gap=0, col=rgb(0, 0, 0, .1))

dat[ `PT08.S3(NOx)` > 0,
     scatter.smooth(`PT08.S5(O3)` ~ `PT08.S3(NOx)`, pch = ".",
                    col="steelblue",
                    lpars = list ( col = "red", lwd = 2, lty=3 ))]
dat_noz <- dat[ `PT08.S3(NOx)` > 0 ]
deg_max <- 10
cols <- rainbow ( deg_max )

x_pred <- seq ( 400, 2600, 10)

for (deg in 1:deg_max) {
  model <- lm(`PT08.S5(O3)` ~ poly(`PT08.S3(NOx)`, deg ),
              data = dat_noz )
  y_pred <- predict ( model, newdata = data.table(`PT08.S3(NOx)`=x_pred) )
  dat_noz [, lines (y_pred ~ x_pred, col=cols[deg], lwd=2 )] 
}

legend ( "topright",
         legend = c ( "loess", paste ( "Poly", 1:deg_max)),
         lwd = 2, lty=c(3, rep(1, deg_max)),
         col=c ("red", rainbow(deg_max)), ncol = 2 )
