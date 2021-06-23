library(nlme)
library(mgcv)
library(rhdf5) # for loading hdf4 
library(rlist)
library(plotfunctions)
library(itsadug)
library(ggplot2)
library(writexl)
library(itsadug)


## Add an alpha value to a colour
add.alpha <- function(col, alpha=1){
  if(missing(col))
    stop("Please provide a vector of colours.")
  apply(sapply(col, col2rgb)/255, 2, 
        function(x) 
          rgb(x[1], x[2], x[3], alpha=alpha))  
}


predict_zones <- function(r_gam){
  # predict all 3 zones speprateley
  x_new<- c(0:495)
  zoneS <- rep('S',496)
  zoneN <- rep('N',496)
  zoneD <- rep('D',496)
  df_newS = data.frame('distance'=x_new, 'zone'=zoneS)
  y_fit_seS <- predict(r_gam, type="response", newdata=df_newS, se.fit = TRUE)
  df_newN = data.frame('distance'=x_new, 'zone'=zoneN)
  y_fit_seN <- predict(r_gam, type="response", newdata=df_newN, se.fit = TRUE)
  df_newD = data.frame('distance'=x_new, 'zone'=zoneD)
  y_fit_seD <- predict(r_gam, type="response", newdata=df_newD, se.fit = TRUE)
  return( list(y_fit_seS,y_fit_seN,y_fit_seD, x_new))
}


plot_zones_predictions<- function(y_fit_seS,y_fit_seN,y_fit_seD, x_new){
  # plot raw mean predicitons and 95% conf interval (assuming normal distr., calculated as 1.96*SE)
  colors_zones = c('red','green','blue')
  
  col=colors_zones[1]
  plot(x_new, y_fit_seS$fit,type='l',col=col, ylim = c(0,9),
       ylab='ribbon density',
       xlab='distance',
       main = 'GAM fits')
  # plot shading
  polygon(c(x_new, rev(x_new)),
          c(y_fit_seS$fit+1.96*y_fit_seS$se.fit,
            rev(y_fit_seS$fit-1.96*y_fit_seS$se.fit)),
          col=add.alpha(col, alpha = 0.5), border = FALSE)
  #lines(x_new, y_fit_seS$fit+1.96*y_fit_seS$se.fit,type='l',lty = 2, lwd = 1,col=col)
  #lines(x_new, y_fit_seS$fit-1.96*y_fit_seS$se.fit,type='l',lty = 2, lwd = 1,col=col)
  
  col=colors_zones[2]
  # plot shading
  polygon(c(x_new, rev(x_new)),
          c(y_fit_seN$fit+1.96*y_fit_seN$se.fit,
            rev(y_fit_seN$fit-1.96*y_fit_seN$se.fit)),
          col=add.alpha(col, alpha = 0.5), border = FALSE)
  lines(x_new, y_fit_seN$fit,type='l',col=col )
  #lines(x_new, y_fit_seN$fit+1.96*y_fit_seN$se.fit,type='l',lty = 2, lwd = 1,col=col)
  #lines(x_new, y_fit_seN$fit-1.96*y_fit_seN$se.fit,type='l',lty = 2, lwd = 1,col=col)
  
  col=colors_zones[3]
  # plot shading
  polygon(c(x_new, rev(x_new)),
          c(y_fit_seD$fit+1.96*y_fit_seD$se.fit,
            rev(y_fit_seD$fit-1.96*y_fit_seD$se.fit)),
          col=add.alpha(col, alpha = 0.5), border = FALSE)
  lines(x_new, y_fit_seD$fit,type='l',col=col )
  #lines(x_new, y_fit_seD$fit+1.96*y_fit_seD$se.fit,type='l',lty = 2, lwd = 1,col=col)
  #lines(x_new, y_fit_seD$fit-1.96*y_fit_seD$se.fit,type='l',lty = 2, lwd = 1,col=col)
  

  legend("topright",
         zones,
         fill=colors_zones)
}





save_diff_plots <- function(r_gam,zone1,zone2,vertical_lines=FALSE, save_jpg=TRUE, save_svg=FALSE, save_eps=FALSE){
  # plot difference
  folderpath_plots = "../gam_comparison/plots/"
  if(save_svg){
    filename = paste(folderpath_plots, "differences_",zone1,'_',zone2,".svg", sep="")
    svg(file = filename)
    out<-plot_diff(r_gam,
                   view='distance',
                   comp=list(zone=c(zone1, zone2)),
                   n.grid=496,
                   mark.diff=vertical_lines
                   )
    x <- find_difference(out$est, out$CI, f=1, xVals=out$distance, as.vector = FALSE)
    
    # add sign diff lines
    for (i in c(1:length(x$start))) {
      segments(x$start[i],0,x$end[i],0,
               col='red',lwd=2 )
    }
    dev.off()
  }
  if(save_jpg){
    filename = paste(folderpath_plots,'differences_',zone1,'_',zone2,".jpeg", sep="")
    jpeg(file = filename, quality = 100)
    out<-plot_diff(r_gam,
                   view='distance',
                   comp=list(zone=c(zone1, zone2)),
                   n.grid=496,
                   mark.diff=vertical_lines)
    x <- find_difference(out$est, out$CI, f=1, xVals=out$distance, as.vector = FALSE)
    
    # add sign diff lines
    for (i in c(1:length(x$start))) {
      segments(x$start[i],0,x$end[i],0,
               col='red',lwd=2 )
    }
    dev.off()
  }
  if(save_eps){
    filename = paste(folderpath_plots,"differences_",zone1,'_',zone2,".eps", sep="")
    #setEPS()
    #postscript(file = filename)
    cairo_ps(file = filename)
    out<-plot_diff(r_gam,
                   view='distance',
                   comp=list(zone=c(zone1, zone2)),
                   n.grid=496,
                   mark.diff=vertical_lines)
    x <- find_difference(out$est, out$CI, f=1, 
                         xVals=out$distance, as.vector = FALSE)
    
    # add sign diff lines
    for (i in c(1:length(x$start))) {
      segments(x$start[i],0,x$end[i],0,
               col='red',lwd=2 )
    }
    dev.off()
  }
  return(out)
}



