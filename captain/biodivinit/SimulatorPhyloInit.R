suppressMessages(library("optparse"))
suppressMessages(library("picante"))
suppressMessages(library("TreeSim"))

# Parsing Command Line Options
option_list <- list(
	make_option("--f"), # out file name 
	make_option("--n", type="integer", default=100), # n tips
	make_option("--i", type="integer", default=100), # rnd ID
	)

parser_object <- OptionParser(option_list=option_list)
opt <- parse_args(parser_object, args = commandArgs(trailingOnly = T), positional_arguments=T)

ntaxa <- opt$options$n


ex_rate <- runif(1,0,0.9)
phylo <- sim.bd.taxa(ntaxa, 1, 1, ex_rate, complete=F)[[1]]



# rename species so the tip order = alphabetic order
tiplab <- c()
for (i in 1:ntaxa){
	if (i<10){
		tp <- paste("T000",i,sep="")
	}else if (i<100){
		tp <- paste("T00",i,sep="")
	}else if (i<1000){
		tp <- paste("T0",i,sep="")
	}else{
		tp <- paste("T",i,sep="")
	}	
	tiplab <- c(tiplab, tp)
}

phylo$tip.label <- tiplab
tbl <- as.data.frame(evol.distinct(phylo))

write.nexus(phylo,file = paste(opt$options$f,opt$options$i,".tre"  ,sep=""))
write.table(tbl  ,file = paste(opt$options$f, opt$options$i, "_ED.txt",sep=""), quote=F, row.names=F, sep="\t")




      
      
