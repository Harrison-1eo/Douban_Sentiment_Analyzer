// ==UserScript==
// @name         Douban Movie Comments Fetcher
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Fetch comments from Douban movie pages
// @author       YourName
// @match        https://movie.douban.com/subject/*
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function() {
    'use strict';

    function fetchAndSendComments() {
        const movieId = window.location.pathname.split('/')[2];
        const commentsUrl = `https://movie.douban.com/subject/${movieId}/comments?start=0&limit=200&status=P&sort=new_score`;

        GM_xmlhttpRequest({
            method: 'GET',
            url: commentsUrl,
            onload: function(response) {
                // Send data to local server
                GM_xmlhttpRequest({
                    method: 'POST',
                    url: 'http://localhost:5000/receive_comments', // Change this to your Python server endpoint
                    headers: {"Content-Type": "application/x-www-form-urlencoded"},
                    data: 'data=' + encodeURIComponent(response.responseText),
                    onload: function(response) {
                        console.log('Data sent to server:', response.responseText);
                    }
                });
            }
        });
    }

    // Create and add button to the page
    // Create and style the button
    const button = document.createElement('button');
    button.textContent = 'Fetch Comments';
    button.style.position = 'fixed';
    button.style.top = '20px';
    button.style.right = '20px';
    button.style.padding = '10px 20px';
    button.style.fontSize = '16px';
    button.style.backgroundColor = '#008CBA';
    button.style.color = 'white';
    button.style.border = 'none';
    button.style.borderRadius = '5px';
    button.style.cursor = 'pointer';
    button.style.zIndex = '1000';
    button.addEventListener('click', fetchAndSendComments);

    // Append the button to the body
    document.body.appendChild(button);
})();
