



%% AM I DOING THINGS RIGHT??

%% CONSTANTS
clear
N = 100;
d = sin((1:N)/10);

lambda = 0.995;

rls_xi = 1e4;
Lmax = 3;
fs = 44100;
fmin = 50;
fmax = 500;
fdist = 50;

%% PEARLS START

% Set to zero if no dictionary updated should be performed
doDictionaryLearning = 1;

% Set to zero if no update speed-up should be performed
doActiveUpdate = 1;

% Nbr of samples for dictionary update
nbrSamplesForPitch = floor(45*1e-3*fs); % use, e.g., 45 ms of the signal

% Settings for speed-up
waitingPeriod = 9e-3; % The waiting period during which a pitch block can be excluded from updating (ms).
blockUpdateThreshold = floor(fs*waitingPeriod); % As above, expressed in nbr of samples.
zeroUpdateThreshold = floor(blockUpdateThreshold/10); % Determine how often to check whether a pitch block should be set to zero
speedUpHorizon = blockUpdateThreshold; % Determine when to start activating/deactivating pitch blocks

% The number of dictyionary length stored in memory
dictionaryLength = 20;

% Initial values for the penalty parameters
gamma = 4;
gamma2 = 80;

% The proximal gradient step-size
stepSize = 1e-4;
maxIter = 20; % maximum number of iterations

% Set to one to print sample number
doPrint = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%
N = length(d);
fpgrid = fmin:fdist:fmax; % the pitch frequency grid
P = length(fpgrid);
nbrOfVariables = P*Lmax;
freqMat = (1:Lmax)'*fpgrid;
t = 0:N-1;
tTemp = 0:dictionaryLength-1;
AInner = 2*pi*tTemp(:)*freqMat(:)'/fs; % original dictionary
AInnerNoPhase = AInner;
A = exp(1i*AInner);
AOld = A;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% WINDOW FOR GAMMA UPDATE %%%%%%%%%%%%%%%%%%%%%
Delta = floor(log(0.01)/log(lambda));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% INITIALIZE ALL VARIABLES %%%%%%%%%%%%%%
xn = A(1,:)';
dn = d(1);
Rn = xn*xn';
rn = xn*dn;
w_hat = zeros(nbrOfVariables,1);

% RLS
w_rls = zeros(nbrOfVariables,1);
w_rls_hist = zeros(nbrOfVariables,N);

fpgridHist = zeros(P,N);

% Counter for block-update
blockNotUpdatedSince = zeros(1,P);
hasBeenUntouchedSince = zeros(1,P);


activeBlocks = 1:P;
inactiveBlocks = [];
nbrActiveBlocks = zeros(N,1);

activeIndices = 1:P*Lmax;
indexMatrix = reshape(activeIndices,Lmax,P);


%% MAIN LOOP

for n=1:N
    if doPrint && mod(n,100)==0
        fprintf('%d av %d\n',n,N)
    end
    nbrActiveBlocks(n) = length(activeBlocks);
    
    %%%%%%%%%%%%%%%%%%%%%%% SAVE PRESENT GRID %%%%%%%%%%%%%%%%%%%%%%
    fpgridHist(:,n) = fpgrid(:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%% NEW SAMPLE %%%%%%%%%%%%%%%%%%%%%%
    samplesLeft = (dictionaryLength-mod(n,dictionaryLength));
    if samplesLeft == dictionaryLength
        sampleIndex = dictionaryLength;
    else
        sampleIndex = mod(n,dictionaryLength);
    end
    xn = A(sampleIndex,:)';
    
    if samplesLeft == dictionaryLength
        AOld = A;
        upperTimeIndex = min(N,(n+dictionaryLength));
        tTemp = t((n+1):upperTimeIndex);
        if upperTimeIndex-n<dictionaryLength
           tTemp = [tTemp,zeros(1,dictionaryLength-(upperTimeIndex-n))];
        end
        AInner = 2*pi*tTemp(:)*freqMat(:)'/fs; % original dictionary
        AInnerNoPhase = AInner;
        A = exp(1i*AInner);
    end
    
    dn = d(n);
    Rn = lambda*Rn + bsxfun(@times, xn,xn');
    rn = lambda*rn + xn*dn;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
    