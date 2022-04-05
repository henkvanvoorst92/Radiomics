
install.packages("PerformanceAnalytics")
install.packages("clipr")
install.packages("rms")
install.packages("Hmisc")
install.packages("dplyr")
install.packages("tableone")
install.packages("mice")
install.packages("psych")
install.packages("brolibrary")
install.packages("tidyr")
install.packages("tidymodels")
install.packages("broom")

library("clipr")
library('MASS')
library(writexl)
library(rms)
library(Hmisc)
library(tidyr)
library(dplyr)
library("PerformanceAnalytics")
library(tableone)
library('readxl')
library(psych)
library(mice)
library(readxl)
library("tidymodels")
library("broom")

# regressions
#DF = read_excel("L:/basic/divi/CMAJOIE/projects/Agnetha/Radiomics/data_and_results/results_7_v2/train_test/test_noscale.xlsx")
DF = read_excel("L:/basic/divi/CMAJOIE/projects/Agnetha/Radiomics/data_and_results/results_robscale_march11/train_test/test_noscale.xlsx")

df <- DF %>%
  mutate(
    mrs_rev = ordered(mrs_rev),
    posttici_c = ordered(posttici_c),
    attempts_to_succes = ordered(attempts_to_succes),
    
    original_shape_Maximum2DDiameterSlice = as.numeric(original_shape_Maximum2DDiameterSlice)/10,
    original_shape_MajorAxisLength = as.numeric(original_shape_MajorAxisLength)/10,
    original_shape_VoxelVolume =  as.numeric(original_shape_VoxelVolume)/100,
    
    original_glcm_Idmn = as.numeric(original_glcm_Idmn)*100,
    original_gldm_DependenceVariance = as.numeric(original_gldm_DependenceVariance)/10,
    original_gldm_LargeDependenceEmphasis = as.numeric(original_gldm_LargeDependenceEmphasis)/100,
    
    
    age = as.numeric(age1),
    sex = as.factor(sex),
    togroin = as.numeric(togroin),
    prev_af = as.numeric(prev_af),
    #occlsegment_c_short = factor(occlsegment_c_short),
    occlsegment_c_short_0 = as.factor(occlsegment_c_short_0),
    occlsegment_c_short_1 = as.factor(occlsegment_c_short_1),
    occlsegment_c_short_2 = as.factor(occlsegment_c_short_2),
    occlsegment_c_short_3 = as.factor(occlsegment_c_short_3),
    occlsegment_c_short_4 = as.factor(occlsegment_c_short_4),
    
    months = as.numeric(DF$months),
    NIHSS_BL = as.numeric(NIHSS_BL),
    
    TICI_2B3 = as.factor(`TICI_2B-3`),
    n_attempt3 = as.factor(`n_attempt_0-3`),
    FPR = as.factor(FPR),
    ThreePR = as.factor(ThreePR),
    GoodmRS = as.factor(GoodmRS)
  )


#outcome = 'mrs_rev'
tmp1 = polr(mrs_rev ~ original_shape_Maximum2DDiameterSlice + 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res1 = tidy(tmp1, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp2 = polr(mrs_rev ~ original_shape_MajorAxisLength+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res2 = tidy(tmp2, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp3 = polr(mrs_rev ~ original_shape_VoxelVolume+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res3 = tidy(tmp3, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

#tmp4 = polr(mrs_rev ~ original_shape_SurfaceArea, Hess = TRUE, data=df)
#res4 = tidy(tmp4, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

######
tmp5 = polr(mrs_rev ~ original_glcm_Idmn + 
              age + sex + prev_af + months + NIHSS_BL + #togroin +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res5 = tidy(tmp5, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 


tmp6 = polr(mrs_rev ~ original_gldm_DependenceVariance+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res6 = tidy(tmp6, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp7 = polr(mrs_rev ~ original_gldm_LargeDependenceEmphasis+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res7 = tidy(tmp7, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

out_mrs = rbind(head(res1,1),head(res2,1),head(res3,1),#head(res4,1),
                head(res5,1),head(res6,1),head(res7,1))

#TICI
tmp1 = polr(posttici_c ~ original_shape_Maximum2DDiameterSlice+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res1 = tidy(tmp1, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp2 = polr(posttici_c ~ original_shape_MajorAxisLength+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res2 = tidy(tmp2, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp3 = polr(posttici_c ~ original_shape_VoxelVolume+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res3 = tidy(tmp3, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

#tmp4 = polr(posttici_c ~ original_shape_SurfaceArea, Hess = TRUE, data=df)
#res4 = tidy(tmp4, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

######
tmp5 = polr(posttici_c ~ original_glcm_Idmn + 
              age + sex + prev_af + months + NIHSS_BL + #togroin +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res5 = tidy(tmp5, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp6 = polr(posttici_c ~ original_gldm_DependenceVariance+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res6 = tidy(tmp6, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp7 = polr(posttici_c ~ original_gldm_LargeDependenceEmphasis+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res7 = tidy(tmp7, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

out_tici = rbind(head(res1,1),head(res2,1),head(res3,1),#head(res4,1),
                 head(res5,1),head(res6,1),head(res7,1))


#TICI attempts_to_succes
tmp1 = polr(attempts_to_succes ~ original_shape_Maximum2DDiameterSlice+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res1 = tidy(tmp1, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp2 = polr(attempts_to_succes ~ original_shape_MajorAxisLength+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res2 = tidy(tmp2, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp3 = polr(attempts_to_succes ~ original_shape_VoxelVolume+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res3 = tidy(tmp3, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

#tmp4 = polr(attempts_to_succes ~ original_shape_SurfaceArea, Hess = TRUE, data=df)
#res4 = tidy(tmp4, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

######
tmp5 = polr(attempts_to_succes ~ original_glcm_Idmn + 
              age + sex + prev_af + months + NIHSS_BL + #togroin +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res5 = tidy(tmp5, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp6 = polr(attempts_to_succes ~ original_gldm_DependenceVariance+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res6 = tidy(tmp6, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

tmp7 = polr(attempts_to_succes ~ original_gldm_LargeDependenceEmphasis+ 
              age + sex + togroin + prev_af + months + NIHSS_BL +
              occlsegment_c_short_1 + occlsegment_c_short_2 + occlsegment_c_short_3 + occlsegment_c_short_4, 
            Hess = TRUE, data=df)
res7 = tidy(tmp7, conf.int = TRUE, conf.level = 0.95,p.value=TRUE, exponentiate = TRUE)#, 

out_attmp = rbind(head(res1,1),head(res2,1),head(res3,1),#head(res4,1),
                  head(res5,1),head(res6,1),head(res7,1))

write_clip(out_mrs) 
write_clip(out_tici) 
write_clip(out_attmp)


