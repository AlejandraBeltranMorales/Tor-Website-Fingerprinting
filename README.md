# Tor Website Fingerprinting

The objective is to deal with the de-anonymization of websites that were visited by the user via the Tor network. We developed a classifier that can predict the visited website based on the observed encrypted network traffic.

We have two codes, one for a closed world and the other for a open world scenario.

## Closed World
This scenario limits the user, as it can only visit certain websites. These websites are known to the attacker, who then tries to reconstruct which sites the user has visited based on certain characteristics in the encrypted network traffic. The training and testing data are confidential, the only relevant piece of information is that they are PCAP files. The output will be the .pcap file followed by its prediction of the name of the website it belongs to.

## Open World
In this scenario, it is assumed that the user can visit any website. The attacker now wants to find out whether the user has visited censored websites. The classifier must know how to distinguish between censored and uncensored websites. Using the close world scenario as the censored, and the training data as the uncensored websites, we must be able to identify correcly in the test data if the website is censored (1) or uncensored (0).
