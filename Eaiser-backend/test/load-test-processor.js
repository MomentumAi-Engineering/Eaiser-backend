
// Load test processor for custom functions
module.exports = {
  // Generate random email
  randomEmail: function(context, events, done) {
    const domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'test.com'];
    const domain = domains[Math.floor(Math.random() * domains.length)];
    const username = 'user' + Math.floor(Math.random() * 100000);
    context.vars.randomEmail = username + '@' + domain;
    return done();
  },
  
  // Generate random full name
  randomFullName: function(context, events, done) {
    const firstNames = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Anna'];
    const lastNames = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller', 'Taylor'];
    const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
    const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
    context.vars.randomFullName = firstName + ' ' + lastName;
    return done();
  },
  
  // Generate random phone number
  randomPhoneNumber: function(context, events, done) {
    const phoneNumber = '+91' + Math.floor(Math.random() * 9000000000 + 1000000000);
    context.vars.randomPhoneNumber = phoneNumber;
    return done();
  },
  
  // Log response times for analysis
  logResponse: function(requestParams, response, context, ee, next) {
    if (response.timings) {
      console.log('Response time:', response.timings.response, 'ms');
    }
    return next();
  },
  
  // Custom error handling
  handleError: function(requestParams, response, context, ee, next) {
    if (response.statusCode >= 400) {
      console.error('Error:', response.statusCode, response.body);
    }
    return next();
  }
};
