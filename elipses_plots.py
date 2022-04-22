#Elipses plots


def plot_ellipse(x,y,xerr,yerr,title='October - November',corr='no',x_label='Eastern Pacific Warming [K K$^{-1}$]',y_label='Central Pacific Warming [K K$^{-1}$]'):
    #Compute regression y on x
    x1 = x.reshape(-1, 1)
    y1 = y.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    reg = linear_regressor.fit(x1, y1)  # perform linear regression
    X_pred = np.linspace(np.min(x)-1, np.max(x)+0.5, 31)
    X_pred = X_pred.reshape(-1, 1)
    Y_pred = linear_regressor.predict(X_pred)  # make predictions
    c = reg.coef_

    #Compute regression x on y
    reg2 = linear_regressor.fit(y1, x1)  # perform linear regression
    Y_pred2 = np.linspace(np.min(y), np.max(y), 31)
    Y_pred2 = Y_pred2.reshape(-1, 1)
    X_pred2 = linear_regressor.predict(Y_pred2)  # make predictions
    c2 = reg2.coef_

    #Define limits
    min_x = np.min(x) - 0.2*np.abs(np.max(x) - np.min(x))
    max_x = np.max(x) + 0.2*np.abs(np.max(x) - np.min(x))
    max_y = np.max(y) + 0.2*np.abs(np.max(y) - np.min(y))
    max_y = np.min(y) - 0.2*np.abs(np.max(y) - np.min(y))
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #Calcular las rectas x = y, x = -y
    Sx = np.std(x)
    Sy = np.std(y)
    S_ratio = Sy/Sx
    YeqX = S_ratio*X_pred - S_ratio*mean_x + mean_y
    YeqMinsX = S_ratio*mean_x + mean_y - S_ratio*X_pred


    #Plot-----------------------------------------------------------------------
    markers = ['<','<','v','*','D','x','x','p','+','+','d','8','X','X','^','d','d','1','2','>','>','D','D','s','.','P', 'P', '3','4']
    #markers = ['o','o','v','8','*','D','D','^','.','.','h','H','<','<','v','8','8','D','X','x','x','p','p','+','_','d', 'd', '>','X','s']
    fig, ax = plt.subplots()
    for px, py, t, l, k, f in zip(x, y, markers, models, xerr, yerr):
       ax.errorbar(px, py, xerr=k, yerr=f,fmt='',ecolor='gray', elinewidth=0.5,capthick=0.001)
       ax.scatter(px, py, marker=t,label=l)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    confidence_ellipse(x, y, ax, corr,edgecolor='red',label='80 $\%$ confidence region')
    confidence_ellipse(x, y, ax,corr,chi_squared=4.6,edgecolor='k',linestyle='--',alpha=0.5,label='$\pm$ 10 $\%$ confidence regions')
    confidence_ellipse(x, y, ax,corr,chi_squared=2.4,edgecolor='k',linestyle='--',alpha=0.5)
    ax.axvline(mean_x, c='grey', lw=1)
    ax.axhline(mean_y, c='grey', lw=1)
    #ax.plot(X_pred,YeqX ,color='black')
    #ax.plot(X_pred,YeqMinsX ,color='grey')
    ax.grid()
    ax.tick_params(labelsize=18)
    if corr == 'yes':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)
        story_x1 = [mean_x + ts1*np.std(x)]
        story_x2 = [mean_x - ts1*np.std(x)]
        story_y_red1 = [mean_y + ts1*np.std(y)]
        story_y_red2 =[mean_y - ts1*np.std(y)]
        story_x3 = [mean_x - ts2*np.std(x)]
        story_x4 = [mean_x + ts2*np.std(x)]
        story_y_red3 = [mean_y + ts2*np.std(y)]
        story_y_red4 =[mean_y - ts2*np.std(y)]
        ax.plot(story_x1, story_y_red1, 'ro',alpha = 0.6,markersize=10,label='storylines')
        ax.plot(story_x2, story_y_red2, 'ro',alpha = 0.6,markersize=10)
        ax.plot(story_x3, story_y_red3, 'ro',alpha = 0.6,markersize=10)
        ax.plot(story_x4, story_y_red4, 'ro',alpha = 0.6,markersize=10)  
    #story_x_70 = [mean_x + 1.09*np.std(x),mean_x - 1.09*np.std(x),mean_x - 1.09*np.std(x),mean_x + 1.09*np.std(x)]
    #print('70 TW',mean_x + 1.09*np.std(x),mean_x - 1.09*np.std(x))
    #story_y_70 = [mean_y + 1.09*np.std(y),mean_y + 1.09*np.std(y),mean_y - 1.09*np.std(y),mean_y - 1.09*np.std(y)]
    #print('70 VB ',mean_y + 1.09*np.std(y),mean_y - 1.09*np.std(y))
    #story_x_90 = [mean_x + 1.515*np.std(x),mean_x - 1.515*np.std(x),mean_x - 1.515*np.std(x),mean_x + 1.515*np.std(x)]
    #print('90 TW',mean_x + 1.515*np.std(x),mean_x - 1.515*np.std(x))
    #print('90 VB ',mean_y + 1.515*np.std(y),mean_y - 1.515*np.std(y))
    #story_y_90 = [mean_y + 1.515*np.std(y),mean_y + 1.515*np.std(y),mean_y - 1.515*np.std(y),mean_y - 1.515*np.std(y)]
    else:
        story_x = [mean_x + 1.26*np.std(x),mean_x - 1.26*np.std(x)]
        story_y_red = [mean_y + 1.26*np.std(y),mean_y + 1.26*np.std(y)]
        story_y_blue =[mean_y - 1.26*np.std(y),mean_y - 1.26*np.std(y)]
        ax.plot(story_x, story_y_red, 'ro',alpha = 0.6,markersize=10,label='High VB (GW)')
        ax.plot(story_x, story_y_blue, 'bo',alpha = 0.6,markersize=10,label='Low VB (GW)')    
    #ax.plot(story_x_70, story_y_70,'ko',alpha=0.5,label='$\pm$ 10$\%$ confidence region storylines')
    #ax.plot(story_x_90, story_y_90,'ko',alpha=0.5)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=5)
    #plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Model')
    #plt.xlim(1.1,2.1)
    #plt.ylim(0,8.5)
    plt.subplots_adjust(bottom=0.05)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    plt.title(title+' R='+str(round(np.corrcoef(x,y)[0,1],3)))
    #plt.savefig(path_plots+'/C_E_index.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.clf
    #plt.clf()
    
def plot_ellipse_seasonal(x,y,xerr,yerr,title='Seasonal Coherence',corr='no',story='yes',x_label='Eastern Pacific Warming [K K$^{-1}$]',y_label='Central Pacific Warming [K K$^{-1}$]'):
    #Compute regression y on x
    x1 = x.reshape(-1, 1)
    y1 = y.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    reg = linear_regressor.fit(x1, y1)  # perform linear regression
    X_pred = np.linspace(np.min(x)-1, np.max(x)+0.5, 31)
    X_pred = X_pred.reshape(-1, 1)
    Y_pred = linear_regressor.predict(X_pred)  # make predictions
    c = reg.coef_

    #Compute regression x on y
    reg2 = linear_regressor.fit(y1, x1)  # perform linear regression
    Y_pred2 = np.linspace(np.min(y), np.max(y), 31)
    Y_pred2 = Y_pred2.reshape(-1, 1)
    X_pred2 = linear_regressor.predict(Y_pred2)  # make predictions
    c2 = reg2.coef_

    #Define limits
    min_x = np.min(x) - 0.2*np.abs(np.max(x) - np.min(x))
    max_x = np.max(x) + 0.2*np.abs(np.max(x) - np.min(x))
    max_y = np.max(y) + 0.2*np.abs(np.max(y) - np.min(y))
    max_y = np.min(y) - 0.2*np.abs(np.max(y) - np.min(y))
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #Calcular las rectas x = y, x = -y
    Sx = np.std(x)
    Sy = np.std(y)
    S_ratio = Sy/Sx
    YeqX = S_ratio*X_pred - S_ratio*mean_x + mean_y
    YeqMinsX = S_ratio*mean_x + mean_y - S_ratio*X_pred


    #Plot-----------------------------------------------------------------------
    markers = ['<','<','v','*','D','x','x','p','+','+','d','8','X','X','^','d','d','1','2','>','>','D','D','s','.','P', 'P', '3','4']
    #markers = ['o','o','v','8','*','D','D','^','.','.','h','H','<','<','v','8','8','D','X','x','x','p','p','+','_','d', 'd', '>','X','s']
    fig, ax = plt.subplots()
    for px, py, t, l, k, f in zip(x, y, markers, models, xerr, yerr):
       ax.errorbar(px, py, xerr=k, yerr=f,fmt='',ecolor='gray', elinewidth=0.5,capthick=0.001)
       ax.scatter(px, py, marker=t,label=l)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    confidence_ellipse(x, y, ax, corr,edgecolor='red',label='80 $\%$ confidence region')
    confidence_ellipse(x, y, ax,corr,chi_squared=4.6,edgecolor='k',linestyle='--',alpha=0.5,label='$\pm$ 10 $\%$ confidence regions')
    confidence_ellipse(x, y, ax,corr,chi_squared=2.4,edgecolor='k',linestyle='--',alpha=0.5)
    ax.axvline(mean_x, c='grey', lw=1)
    ax.axhline(mean_y, c='grey', lw=1)
    #ax.plot(X_pred,YeqX ,color='black')
    #ax.plot(X_pred,YeqMinsX ,color='grey')
    ax.grid()
    ax.tick_params(labelsize=18)
    r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
    ts = np.sqrt(((1-r**2)/(2*(1-r)))*chi)
    print(r,ts)
    story_x1 = [mean_x + ts*np.std(x)]
    story_x2 = [mean_x - ts*np.std(x)]
    story_y_red1 = [mean_y + ts*np.std(y)]
    story_y_red2 =[mean_y - ts*np.std(y)]
    #story_x_70 = [mean_x + 1.09*np.std(x),mean_x - 1.09*np.std(x),mean_x - 1.09*np.std(x),mean_x + 1.09*np.std(x)]
    #print('70 TW',mean_x + 1.09*np.std(x),mean_x - 1.09*np.std(x))
    #story_y_70 = [mean_y + 1.09*np.std(y),mean_y + 1.09*np.std(y),mean_y - 1.09*np.std(y),mean_y - 1.09*np.std(y)]
    #print('70 VB ',mean_y + 1.09*np.std(y),mean_y - 1.09*np.std(y))
    #story_x_90 = [mean_x + 1.515*np.std(x),mean_x - 1.515*np.std(x),mean_x - 1.515*np.std(x),mean_x + 1.515*np.std(x)]
    #print('90 TW',mean_x + 1.515*np.std(x),mean_x - 1.515*np.std(x))
    #print('90 VB ',mean_y + 1.515*np.std(y),mean_y - 1.515*np.std(y))
    #story_y_90 = [mean_y + 1.515*np.std(y),mean_y + 1.515*np.std(y),mean_y - 1.515*np.std(y),mean_y - 1.515*np.std(y)]
    if story == 'yes':
        ax.plot(story_x1, story_y_red1, 'ro',alpha = 0.6,markersize=10,label='High VB (GW)')
        ax.plot(story_x2, story_y_red2, 'ro',alpha = 0.6,markersize=10,label='Low VB (GW)')
    else:
        a = 'nada'
    #ax.plot(story_x_70, story_y_70,'ko',alpha=0.5,label='$\pm$ 10$\%$ confidence region storylines')
    #ax.plot(story_x_90, story_y_90,'ko',alpha=0.5)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=5)
    #plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Model')
    #plt.xlim(1.1,2.1)
    #plt.ylim(0,8.5)
    plt.subplots_adjust(bottom=0.05)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    plt.title(title+' R='+str(round(np.corrcoef(x,y)[0,1],3)))
    #plt.savefig(path_plots+'/C_E_index.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.clf
    #plt.clf()