library(nlme)
library(mgcv)
library(rhdf5) # for loading hdf4 
library(rlist)
#library(rfast)
library(plotfunctions)
library(itsadug)
library(ggplot2)
library(writexl)
library(itsadug)
# load utils:
source("utils.R")
#library(gsubfn)

##############################


## load data
folderpath='../data/'
filename_led = paste(folderpath, 'vesicle_densities.csv', sep='' )
df <-read.csv(filename_led, header=TRUE,sep=',',dec='.')
df<- subset(df, select=-c(X))

# fit model
r_gam<-gam(density~s(distance, by=zone,k=100)+zone, #bs="cr"
           data=df, 
           gamma=1, 
           family = gaussian
           )

# predict zones
y_fit_se <- predict_zones(r_gam)
y_fit_seS <- y_fit_se[1][[1]]
y_fit_seN <- y_fit_se[2][[1]]
y_fit_seD <- y_fit_se[3][[1]]
x_new <- y_fit_se[4][[1]]


# plot predictions
plot_zones_predictions(y_fit_seS,y_fit_seN,y_fit_seD,x_new)

# show gam summary
summary(r_gam)

# print details to .txt file
sink("gam_summary.txt", append = TRUE)
print(summary(r_gam))
print('ANOVA on GAM: anova.gam results:')
print(anova.gam(r_gam))
print('------')
sink() 


# plot differences for specific comparison
out<-plot_diff(r_gam,
          view='distance',
          comp=list(zone=c('S','D') ),
          n.grid=496,
          mark.diff=FALSE # add vertical lines if TRUE
          )
x <- find_difference(out$est, out$CI, f=1, xVals=out$distance, as.vector = FALSE)
# add sign diff lines
for (i in c(1:length(x$start))) {
  segments(x$start[i],0,x$end[i],0,
           col='red',lwd=2 )
}


# plot and save all differences
zones = c('S','N','D')
for (zone1 in zones){
  for (zone2 in subset(zones, !zones==zone1)){
    out = save_diff_plots(r_gam,zone1,zone2,save_jpg=TRUE, save_svg=FALSE, save_eps=TRUE)
    # find differences 
    x <- find_difference(out$est, out$CI, f=1, xVals=out$distance, as.vector = FALSE)
    # store to dataframe
    if(zone1=='S' & zone2=='N'){
      df_results<-data.frame( 'zone1'=zone1, 
                              'zone2'=zone2, 
                              'sign_difference_start'=x$start,
                              'sign_difference_end'=x$end,
                              stringsAsFactors = FALSE)
    }else{
      for(j in c(1:length(x$start))){
        newrow = c(zone1, zone2,x$start[j],x$end[j])
        df_results<-rbind(df_results, newrow)
      }
    }
  }
}



# plot and save predictions
folderpath_plots = "../gam_comparison/plots/"
# as jpg
filename = paste(folderpath_plots,"predictions_zones_v2.jpeg", sep="")
jpeg(file = filename, quality = 100)
plot_zones_predictions(y_fit_seS,y_fit_seN,y_fit_seD, x_new)
dev.off()
# as eps
filename = paste(folderpath_plots,"predictions_zones_v2.eps", sep="")
cairo_ps(file = filename)
plot_zones_predictions(y_fit_seS,y_fit_seN,y_fit_seD, x_new)
dev.off()
 

# plot predictions one by one

plot_onezone_prediction<- function(y_fit_se, filename_raw,folderpath_plots,colornr=1){
  
  colors_zones = c('red','green','blue')
  
  col=colors_zones[colornr]
  # as jpg
  filename_raw2 = paste(filename_raw,'.jpg', sep="")
  filename = paste(folderpath_plots,filename_raw2, sep="")
  jpeg(file = filename, quality = 100)
  plot(x_new, y_fit_se$fit,type='l',col=col, ylim = c(0,9),
       ylab='ribbon density',
       xlab='distance',
       main = 'GAM fits')
  # plot shading
  polygon(c(x_new, rev(x_new)),
          c(y_fit_se$fit+1.96*y_fit_se$se.fit,
            rev(y_fit_se$fit-1.96*y_fit_se$se.fit)),
          col=add.alpha(col, alpha = 0.5), border = FALSE)
  dev.off()
  
  # as eps
  filename_raw2 = paste(filename_raw,'.eps', sep="")
  filename = paste(folderpath_plots,filename_raw2, sep="")
  cairo_ps(file = filename)
  plot(x_new, y_fit_se$fit,type='l',col=col, ylim = c(0,9),
       ylab='ribbon density',
       xlab='distance',
       main = 'GAM fits')
  # plot shading
  polygon(c(x_new, rev(x_new)),
          c(y_fit_se$fit+1.96*y_fit_se$se.fit,
            rev(y_fit_se$fit-1.96*y_fit_se$se.fit)),
          col=add.alpha(col, alpha = 0.5), border = FALSE)
  dev.off()
}

# set folderpath
folderpath_plots = "../gam_comparison/plots/"

#choose zone
y_fit_se = y_fit_seD
filename_raw ='prediction_zone_D'
colornr=3
# plot
plot_onezone_prediction(y_fit_se, filename_raw,folderpath_plots,colornr=colornr)

 
# write to excel file
write_xlsx(df_results,"gam_results_test.xlsx")

