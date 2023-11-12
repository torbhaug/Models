%% Editors note
% There mght be some insances where the variable n_runs  get really large
% and causes the zeros() function to error to fix this either manage
% allowed use of RAM in your settings or attempt fewer runs
% The settings can be found under the Home tab in
% Preferences/Workspace/MatLab array size limits

%% Creating a pair of patterns
% Creating 2 arrays of binary 1/-1 state information

N = 50; % Number of neurons

% Creating T and noise variables
T  = 200; % total time
noise = 0.2; % setting a level of noise for the simulation
noisy_n = round(N * noise); % setting how many values that will be replaced by noise
%t = 1; % current time

patterns = nan(N, 2);
for p = 1:2
    patterns(:, p) 
end

% Precreating df of zeros
W = zeros(50, 50);

% Creating m variables for both patterns
m_v = nan(T, 1); % overlap with pattern v at each time
m_u = nan(T, 1); % overlap with pattern u at each time
% To find the final match display m_*(T)

% Creating 2 random patterns
% Pattern 1
% Generating an array of 50 random 1 or -1  values
v = randi([0 1], 1, N);
v = changem(v, -1, 0);
disp(v)

% Pattern 2
% Generating an array of 50 random 1 or -1  values
u = randi([0 1], 1, N);
u = changem(u, -1, 0);
disp(u)

% Creating a pattern having stored both patterns
for i = 1:numel(v)
    %index_i = index_i + 1;
    for j = 1:numel(v)
        %index_j = index_j + 1;
        if i ~= j
            %temp_vi = v(index_i)
            %temp_vj = v(index_j)
            %W(index_i, index_j) = changem(W(index_i, index_j), i * j);
            W(i, j) = (v(i) * v(j)) + (u(i) * u(j)) / N;
        end
    end
end 

%% Creating an input pattern by adding noise to pattern v

input_pat = v; % Precreating an input pattern
outcome_pat = nan; % Precreating an outcome variable

% The next task is to generate a randomized way of drawing for the pattern
% Creating a list of integers in range 1:N counting upwards from 1
int_list = [];
for int = 1:N
    int_list(end+1) = int;
end

% Randomizing the list
    int_list = int_list(randperm(length(int_list))); 
    
% Setting the input pattern as S and then changing S into the noisy
% vesion of the input pattern
S = input_pat;

for i = 1:noisy_n
% selecting numbers of noisy neurons
    if rand <= 0.5
        S(int_list(i)) = -1;
    else
        S(int_list(i)) = 1;
    end
end

% Creating m variables for both pattern as it runs through the 
% internal dynamics of the network
match_v = nan(T, 1); % overlap with pattern v at each time
match_u = nan(T, 1); % overlap with pattern u at each time
% To access the initial pattern take out m_*(1)
% To access the final pattern take out m_*(T)

% Calculating the match for the stored pattern and run through the
% internal network dynamics
for num = 1:T
    match_v(num) = (sum(S .* v)) / N; % overlap for for state with pattern v
    match_u(num) = (sum(S .* u)) / N; % overlap for for state with pattern u
    h = W * S'; % inner product of weights and states
    s = randi(N); %taking a random neuron from S
    S(s) = sign(h(s)); % update state of random neuron
end

disp(match_v)
disp(match_u)

%% The step involves making a loop which  runs through a series of loops of
% the general code to storing the hit rate and the run time of each run
% the variables are brought outside of the base code to be varied in
% differente tests variations
N = 50;
n_run = 10000; % Number of runs the simulation makes
noise_level = 0.2;
T = 200; % The maximum number of iterations the  system is allowed
run_time = zeros(n_run, 1);
hit_list = zeros(n_run, 1);


for run = 1:n_run
    % Creating T and noise variables
    noise = noise_level; % setting a level of noise for the simulation
    noisy_n = round(N * noise); % setting how many values that will be replaced by noise
    
    %  Creating a pair of patterns
    patterns = nan(N, 2);
    for p = 1:2
        patterns(:, p) 
    end
    
    % Precreating df of zeros
    W = zeros(50, 50);

    % Precreating m variables for both patterns
    m_v = nan(T, 1); % overlap with pattern v at each time
    m_u = nan(T, 1); % overlap with pattern u at each time
    % To find the final match display m_*(T)

    % Creating 2 random patterns
    % Pattern 1
    % Generating an array of 50 random 1 or -1  values
    v = randi([0 1], 1, N);
    v = changem(v, -1, 0);

    % Pattern 2
    % Generating an array of 50 random 1 or -1  values
    u = randi([0 1], 1, N);
    u = changem(u, -1, 0);
    disp(u)

    % Creating a pattern having stored both patterns
    for i = 1:numel(v)
        %index_i = index_i + 1;
        for j = 1:numel(v)
            %index_j = index_j + 1;
            if i ~= j
                %temp_vi = v(index_i)
                %temp_vj = v(index_j)
                %W(index_i, index_j) = changem(W(index_i, index_j), i * j);
                W(i, j) = (v(i) * v(j)) + (u(i) * u(j)) / N;
            end
        end
    end 

    input_pat = v; % Precreating an input pattern
    outcome_pat = nan; % Precreating an outcome variable
    
    % The next task is to generate a randomized way of drawing for the pattern
    % Creating a list of integers in range 1:N counting upwards from 1
    int_list = [];
    for int = 1:N
        int_list(end+1) = int;
    end
    
    % Randomizing the list
    int_list = int_list(randperm(length(int_list))); 
        
    % Setting the input pattern as S and then changing S into the noisy
    % vesion of the input pattern
    S = input_pat;
    
    for i = 1:noisy_n
    % selecting numbers of noisy neurons
        if rand <= 0.5
            S(int_list(i)) = -1;
        else
            S(int_list(i)) = 1;
        end
    end
    
    % Saving patterns to file
    %save('Basic_Test_Pats', "v", "u", "S")

    % Loading patterns from file
    % load('Basic_Test_Pats.mat')

    % Creating m variables for both pattern as it runs through the 
    % internal dynamics of the network
    match_v = zeros(T, 1); % overlap with pattern v at each time
    match_u = zeros(T, 1); % overlap with pattern u at each time
    % To access the initial pattern take out m_*(1)
    % To access the final pattern take out m_*(T)
    
    % Calculating the match for the stored pattern and run through the
    % internal network dynamics
    for num = 1:T
        match_v(num) = (sum(S .* v)) / N; % overlap for for state with pattern v
        if and(match_v(num) == 1, run_time(run) == 0) % storing the T for which the pattern stabilizes
            run_time(run) = num;
            hit_list(run) = 1;
        end
        match_u(num) = (sum(S .* u)) / N; % overlap for for state with pattern u
        if and(match_u == 1, run_time(run) == 0) % storing the T for which the pattern stabilizes
            run_time(run) = num;
        end
        h = W * S'; % inner product of weights and states
        s = randi(N); %taking a random neuron from S
        S(s) = sign(h(s)); % update state of random neuron
    end
end

% Adjusting the run time variable so that the runs without centering on
% a pattern gains the maximum runtime value by changing the zeros to T
run_time = changem(run_time, T, 0);

% Creating descriptive variables from the simulation
hit_rate = sum(hit_list)/n_run
% %  Creating a plot for the  hit-list
% plot(hit_list(1:n_run), 'b+')
% hold on
% legend('Hit Along the Runs')
% xlabel('Runs')
% ylabel('Hit')
% hold off

avg_run_time = sum(run_time)/n_run
% Creating a plot for the runtime
runtime_bar = zeros(1:n_run);
runtime_bar = changem(runtime_bar, avg_run_time, 0);

run_time = sort(run_time, 1)
plot(run_time(1:n_run), 'b', LineWidth=1.5)
hold on
plot(runtime_bar(1:n_run), LineWidth=1.5)
legend('Running Times', 'Average Running Time')
xlabel('Runs'), ylabel('Running Times (T)')
hold off


time_variance = (sum(run_time) - avg_run_time)*2/n_run

%% Setting the variables for the different tests
% Initially creating the variables  as before
% Notable exception is the lrn_cost variable that scales the number of
% updates with a learning cost constant
% Noise level has also been changed to pattern_noise for generating the
% pattern with noise level relating to the noise inside the modelled runs

% By changing the variables in this window you can manipulated the
% modelled runs
N = 50;  % The number of neurons in the hopfield networks
n_run = 10000; % Number of runs the simulation makes
pat_noise = 0.2; % Noise added to pattern V to make pattern M
noise_level = 0.2; % Noise added in the simulation
T = 200; % Maximum number of iterations given to the function
max_runtime = 200;
%run_time = zeros(n_run, 1);
%hit_list = zeros(n_run, 1);
%run_var = 1; % the number of variations of the model you want to compare


% Cost constants
LC = 1; % Learning
HRC = 1; % Hit-rate
TC = 1; % Iterations/Time


% Total cost variables
%lrn_cost = zeros(n_run, run_var); % The learncost array
%precision_cost = zeros(n_run, run_var);
%running_cost = zeros(n_run, run_var);

%% Generating the 3 stable patterns 

%This section creates the pattern

% These are to be kept the same for the analysisunless changed by the
% learning rule
% Creating 2 random patterns
% Pattern 1
% Generating an array of 50 random 1 or -1  values
v = randi([0 1], 1, N);
v = changem(v, -1, 0);
disp(v)

% Pattern 2
% Generating an array of 50 random 1 or -1  values
u = randi([0 1], 1, N);
u = changem(u, -1, 0);
disp(u)

% Creating a new input pattern by varying V over noise
input_pat = v; % Precreating an input pattern
outcome_pat = nan; % Precreating an outcome variable
        
% The next task is to generate a randomized way of drawing for the pattern
% Creating a list of integers in range 1:N counting upwards from 1
int_list = [];
for int = 1:N
    int_list(end+1) = int;
end
        
% Randomizing the list
int_list = int_list(randperm(length(int_list))); 
% Setting the input pattern as S and then changing S into the noisy
% version of the input pattern
noisy_n = N * pat_noise;
for i = 1:noisy_n
% selecting numbers of noisy neurons
    if rand <= 0.5
        input_pat(int_list(i)) = -1;
    else
        input_pat(int_list(i)) = 1;
    end
end

% Saving the patterns to file
% save('Extended_Testing_Pats', "v", "u", "input_pat")

%% Loading stored pattern
load('Extended_Testing_Pats.mat')

%% Now  introducing a learning rule that updates the primary stored pattern
% This  function will take the statistical average of the input pattern as
% the model runs through its internal dynamics

total_cost = 0; % Variable for keeping track of the total cost of the model

% Calling the generalized hopfield loop function with the number 1
% indicating n_run/1 aka. running all the indicated number of 
% runs in this particular function
[hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 1, N, T, noise_level, v, u, input_pat)


x = hit_rate;
y = avg_run_time;
z = time_variance;
q = avg_S;

% Tallying number of learning updates and creating descriptive variables
lrn_updates = 0
precison_cost = n_run * (1 - hit_rate) * HRC;
avg_precision_cost = (1 - hit_rate) * HRC
running_cost = n_run * avg_run_time * TC;
avg_running_cost = avg_run_time * TC
learning_cost = lrn_updates * LC

total_cost = total_cost + precison_cost + running_cost + learning_cost
avg_cost = total_cost/n_run

%% Running the model as it the input varies
% Uncomment the variables you want to vary in the first code within the
% loop
variations = 10; % Number of variations of a variable the model will run through


total_cost = zeros(variations, 1); % The total cost for different variations

%NB. For discernable results only vary one of these variables at the time

% Precreating list of variables
noise_list = zeros(variations, 1);
TC_list = zeros(variations, 1);
HRC_list = zeros(variations, 1);
LC_list = zeros(variations, 1);

for var = 1:variations
    noise_list(var) = var * 0.1 % noise_level
    TC_list(var) = TC %var * 0.5
    HRC_list(var) = HRC %var * 0.5
    LC_list(var) = LC %var * 05
    % Take away the constant cost and add varying cost to use

    % Calling the generalized hopfield loop function with the number 1
    % indicating n_run/1 aka. running all the indicated number of 
    % runs in this particular function

    % Specifically if  you are testing noise uncomment underneath line
    noise_level = noise_list(var);

    [hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 1, N, T, noise_level, v, u, input_pat)
    
    
    x = hit_rate;
    y = avg_run_time;
    z = time_variance;
    q = avg_S;
    
    % Creating cost variables
    lrn_updates = 0; % Tallying number of learning updates
    precision_cost = (1-hit_rate) * HRC_list(var)
    running_cost = avg_run_time * TC_list(var)
    learning_cost = lrn_updates * LC_list(var)
    
    total_cost(var) = total_cost(var) + precision_cost + running_cost + learning_cost
end

% Creating descriptive variables
mean_cost = mean(total_cost) % Mean value
max_cost = max(total_cost) % Highest value
min_cost = min(total_cost) % Lowest value

% Creating a stored variable for the sake of comparing to the learning rule
% later
single_run_total_cost = total_cost;
% Run the variation with learning code directly after this to display
% them together

%% This code will run the model once returning the range and mean of the cost

total_cost = 0; % Variable for keeping track of the total cost of the model

% Calling the generalized hopfield loop function with the number 1
% indicating n_run/1 aka. running all the indicated number of 
% runs in this particular function
[hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 2, N, T, noise_level, v, u, input_pat)

lrn_updates = 0;
precison_cost = hit_rate * HRC;
running_cost = avg_run_time * TC;
learning_cost = lrn_updates * LC;

total_cost = sum(total_cost + precison_cost + running_cost + learning_cost)

% Running the function with pattern v adjusted to the average
% individual input
[hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 2, N, T, noise_level, avg_S, u, input_pat)

% Tallying number of learning updates and creating descriptive variables
lrn_updates = 1;
precison_cost = n_run * (1 - hit_rate) * HRC;
avg_precision_cost = (1 - hit_rate) * HRC
running_cost = n_run * avg_run_time * TC;
avg_running_cost = avg_run_time * TC
learning_cost = lrn_updates * LC

total_cost = total_cost + precison_cost + running_cost + learning_cost
avg_cost = total_cost/n_run

%% This code will vary a variable as it runs through the function twice with an update
% Uncomment the variable you intend to vary

variations = 10; % Number of variations of a variable the model will run through

total_cost = zeros(variations, 1); % The total cost for different variations

%NB. For discernable results only vary one of these variables at the time

% Precreating list of variables
pat_noise_list = zeros(variations, 1);
TC_list = zeros(variations, 1);
HRC_list = zeros(variations, 1);
LC_list = zeros(variations, 1);

for var = 1:variations
    noise_list(var) = var * 0.1 % noise_level
    TC_list(var) = var * 0.5 % TC
    HRC_list(var) = HRC % var * 0.5
    LC_list(var) = LC %var * 05

    % Calling the generalized hopfield loop function with the number 1
    % indicating n_run/1 aka. running all the indicated number of 
    % runs in this particular function

    % Specifically if  you are testing noise uncomment underneath line
    noise_level = noise_list(var);

    [hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 2, N, T, noise_level, v, u, input_pat)
    
    
    x = hit_rate;
    y = avg_run_time;
    z = time_variance;
    q = avg_S;
    
    % Creating cost variables
    lrn_updates = 0; % Tallying number of learning updates
    precision_cost = (1-hit_rate) * HRC_list(var)
    running_cost = avg_run_time * TC_list(var)
    learning_cost = lrn_updates * LC_list(var)

    total_cost(var) = total_cost(var) + precision_cost + running_cost + learning_cost;
    
    % Running the function with pattern v adjusted to the average
    % individual input
    [hit_rate, avg_run_time, time_variance, avg_S] = Hopfield_Function(n_run, 2, N, T, noise_level, avg_S, u, input_pat)
    
    % Creating cost variables
    lrn_updates =  1; % Creating a variable to reflect the number of updates made
    precision_cost = (1-hit_rate) * HRC_list(var);
    running_cost = avg_run_time * TC_list(var);
    learning_cost = lrn_updates * LC_list(var);
    
    
    total_cost(var) = (total_cost(var) + precision_cost + running_cost + learning_cost)/2
end

% Creating descriptive variables
mean_cost = mean(total_cost) % Mean value
max_cost = max(total_cost) % Highest value
min_cost = min(total_cost) % Lowest value

% Creating a plot for the total cost of  the model as the variable varies
% Insert the correct variable in the first parameter of the plot function
plot(noise_list, total_cost, LineWidth=1.5, Color='blue')
hold on
plot(noise_list, single_run_total_cost, LineWidth=1.5, Color='red', LineStyle='--')
legend('Cost w. Learning', 'Cost w.out Learning')
% Insert the name of the relevant variable in the xlabel brackets
xlabel('Noise'), ylabel('Total Cost')
hold off
