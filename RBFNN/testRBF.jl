function testRbf(Centers, betas, Theta, nn, X)
	rbf_activations = getRBFActivations(Centers, betas, X)
	model = nn(Theta)
	nn_logits_ = model(rbf_activations)
	return softmax(nn_logits_; dims=1)
end