close all
clear all
todo=[3];
osave = true;
indir='./';
cax = [0 15];
if (sum(todo==1 | todo==2 |todo==3 )>0 )
  Hfil = ncread([indir 'state_true.nc'],'Hfil');
end %fig 1,2

if(sum(todo==1)>0)
  figure(1);
  clf
  imagesc(Hfil(:,:,1)',cax);
  axis image
  colorbar
  if osave
    print -dpng -r300 fig1.png
  end
  
end %fig 1



if(sum(todo==2)>0)
  figure(2);
  clf
  imagesc(Hfil(:,:,end)',cax);
  axis image
  colorbar
  if osave
    print -dpng -r300 fig2.png
  end
  
end %fig 2



if(sum(todo==3)>0)
  figure(3);
  clf
  i0=50;
  j0=60;
  plot(squeeze(Hfil(i0,j0,:)),'m','LineWidth',2);
  xlabel('temps');
  ylabel('h(t)');
  
  if osave
    print -dpng -r300 fig3.png
  end
  
end %fig 2


