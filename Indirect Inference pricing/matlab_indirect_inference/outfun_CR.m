function stop = outfun_CR(x, optimvalues, state)

global logfile
fid = fopen(logfile,'a');
fprintf(fid,'Iteration: %5.0f fcount: %5.0f chi2: %11.6f ',optimvalues.iteration,optimvalues.funccount,optimvalues.fval);
fprintf(fid,'\n');
fprintf(fid,' %8.5f ',x(:));
fprintf(fid,'\n');
fclose(fid);
stop=false;
